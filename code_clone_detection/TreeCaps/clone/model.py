import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from Clone.TreeCaps.clone.utils import softmax,reduce_sum
cuda1 = torch.device('cuda:0')
def truncated_normal_(tensor, mean=0, std=0.09):  # https://zhuanlan.zhihu.com/p/83609874  tf.trunc_normal()

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)


    out_shape = orig_shape[:-1] + list(params.shape)[m:]


    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()

def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    # children is shape (batch_size x max_tree_size x max_children)
    batch_size = children.size(0)
    max_tree_size = children.size(1)
    max_children = children.size(2)
    # eta_t is shape (batch_size x max_tree_size x max_children + 1)
    return (torch.unsqueeze(torch.cat(
        [torch.ones((max_tree_size, 1)).to(children.device), torch.zeros((max_tree_size, max_children)).to(children.device)],
        dim=1), dim=0,
        )).repeat([batch_size, 1, 1])

def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belogs to the 'right'"""
    children = children.type(torch.float32)
    batch_size = children.size(0)
    max_tree_size = children.size(1)
    max_children = children.size(2)

    # num_siblings is shape (batch_size x max_tree_size x 1)
    num_siblings = torch.sum((~(children == 0)).int(),dim=2,keepdim=True,dtype=torch.float32)
    # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
    num_siblings = num_siblings.repeat(([1, 1, max_children + 1]))

    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        [torch.zeros((batch_size, max_tree_size, 1)).to(children.device),
         torch.min(children, torch.ones((batch_size, max_tree_size, max_children)).to(children.device))],
        dim=2)
    # child indices for every tree (batch_size x max_tree_size x max_children + 1)
    child_indices = torch.mul(
        (torch.unsqueeze(
            torch.unsqueeze(
                # torch.arange(-1.0, max_children.type(torch.float32),1.0, dtype=torch.float32),
                torch.arange(-1.0, torch.tensor(max_children, dtype=torch.float32, device=cuda1),1.0, dtype=torch.float32),
                dim=0),
        dim=0).repeat([batch_size, max_tree_size, 1])).cuda(),
        mask
    )

    # weights for every tree node in the case that num_siblings = 0
    # shape is (batch_size x max_tree_size x max_children + 1)
    singles = torch.cat(
        [torch.zeros((batch_size, max_tree_size, 1)).to(children.device),
         torch.full((batch_size, max_tree_size, 1), 0.5).to(children.device),
         torch.zeros((batch_size, max_tree_size, max_children - 1)).to(children.device)],
        dim=2)

    # eta_r is shape (batch_size x max_tree_size x max_children + 1)
    return torch.where(
        # torch.equal(num_siblings, 1.0),
        torch.eq(num_siblings, 1.0),
        # avoid division by 0 when num_siblings == 1
        singles,
        # the normal case where num_siblings != 1
        (1.0 - t_coef) * (child_indices / (num_siblings - 1.0))
    )

def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    children = children.type(torch.float32)
    batch_size = children.size(0)
    max_tree_size = children.size(1)
    max_children = children.size(2)

    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        [torch.zeros((batch_size, max_tree_size, 1)).to(children.device),
         torch.min(children, torch.ones((batch_size, max_tree_size, max_children)).to(children.device))],
        dim=2)

    # eta_l is shape (batch_size x max_tree_size x max_children + 1)

    return torch.mul(
        torch.mul((1.0 - coef_t), (1.0 - coef_r)),mask
    )

def children_tensor( nodes, children, feature_size):
    max_children = torch.tensor(children.size(2))
    batch_size = torch.tensor(nodes.size(0))
    num_nodes = torch.tensor(nodes.size(1))

    # replace the root node with the zero vector so lookups for the 0th
    # vector return 0 instead of the root vector
    # zero_vecs is (batch_size, num_nodes, 1)
    zero_vecs = torch.zeros((batch_size, 1, feature_size),device=cuda1)
    # vector_lookup is (batch_size x num_nodes x feature_size)
    vector_lookup = torch.cat([zero_vecs, nodes[:, 1:, :]], dim=1)
    # children is (batch_size x num_nodes x num_children x 1)
    children = torch.unsqueeze(children, dim=3)
    # prepend the batch indices to the 4th dimension of children
    # batch_indices is (batch_size x 1 x 1 x 1)
    batch_indices = torch.reshape(torch.arange(0, batch_size), (batch_size, 1, 1, 1)).cuda()
    batch_indices = batch_indices.repeat([1, num_nodes, max_children, 1])
    # batch_indices is (batch_size x num_nodes x num_children x 1)        batch_indices = batch_size.repeat(1, num_nodes, max_children, 1)
    # children is (batch_size x num_nodes x num_children x 2)
    children = torch.cat([batch_indices, children], dim=3)
    # output will have shape (batch_size x num_nodes x num_children x feature_size)
    return gather_nd(vector_lookup, children)

def conv_step(nodes, children,feature_size,w_t, w_l, w_r, b_conv):
    # nodes is shape (batch_size x max_tree_size x feature_size)
    # children is shape (batch_size x max_tree_size x max_children)

    # children_vectors will have shape
    # (batch_size x max_tree_size x max_children x feature_size)
    children_vectors = children_tensor(nodes, children, feature_size)

    # add a 4th dimension to the nodes tensor
    nodes = torch.unsqueeze(nodes, 2)

    # tree_tensor is shape
    # (batch_size x max_tree_size x max_children + 1 x feature_size)
    tree_tensor = torch.cat([nodes, children_vectors], dim=2)

    # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
    c_t = eta_t(children)

    c_r = eta_r(children, c_t)
    c_l = eta_l(children, c_t, c_r)

    # concatenate the position coefficients into a tensor
    coef = torch.stack([c_t, c_r, c_l], dim=3)
    weights = torch.stack([w_t, w_r, w_l], dim=0)

    batch_size = children.size(0)
    max_tree_size = children.size(1)
    max_children = children.size(2)

    # reshape for matrix multiplication
    x = batch_size * max_tree_size
    y = max_children + 1

    result = tree_tensor.reshape(x, y, feature_size)
    coef = coef.reshape(x, y, 3)
    result = torch.matmul(result.transpose(1, 2), coef)
    result = torch.reshape(result, (batch_size, max_tree_size, 3, feature_size))

    result = torch.tensordot(result, weights, [[2, 3], [0, 1]])

    return torch.tanh(result + b_conv)


def pool_layer(nodes):
    """Creates a max dynamic pooling layer from the nodes."""
    pooled = torch.max(nodes, 1)
    return pooled.values

def squash(vector):
    vec_squared_norm = reduce_sum(torch.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)


def vts_routing(input, top_a, top_b, num_outputs, num_dims):
    # alpha_IJ = tf.zeros((int(num_outputs / top_a * top_b), num_outputs), dtype=tf.float32)
    alpha_IJ = torch.zeros((int(num_outputs / top_a * top_b), num_outputs), dtype=torch.float32,device=cuda1)

    input = input.permute(0, 2, 3, 1)
    u_i, _ = torch.topk(input, k=top_b)
    u_i = u_i.permute(0, 3, 1, 2)
    u_i = torch.reshape(u_i, (-1, num_dims))
    # u_i = tf.stop_gradient(u_i)
    u_i = u_i

    input, _ = torch.topk(input, k=top_a)
    input = input.permute(0, 3, 1, 2)
    v_J = input
    v_J = torch.reshape(v_J, (-1, num_dims))

    # for rout in range(1):
    u_produce_v = torch.matmul(u_i, v_J.transpose(0,1))
    alpha_IJ += u_produce_v
    beta_IJ = softmax(alpha_IJ, axis=-1)
    v_J = torch.matmul(beta_IJ.transpose(0, 1), u_i)

    v_J = torch.reshape(v_J, (1, num_outputs, num_dims, 1))
    return squash(v_J)


def dynamic_routing(shape, input, W, biases, num_outputs=10, num_dims=16):
    """The Dynamic Routing Algorithm proposed by Sabour et al."""

    iter_routing = 3
    input_shape = shape
    # W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
    #                     dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
    # biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    delta_IJ = torch.zeros([input_shape[0], input_shape[1], num_outputs, 1, 1], dtype=torch.float32,device=cuda1)


    input = input.repeat([1, 1, num_dims * num_outputs, 1, 1])

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = torch.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    u_hat_stopped = u_hat
    # gamma_IJ = softmax(delta_IJ, axis=2)
    # s_J = torch.mul(gamma_IJ, u_hat)
    # s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
    # v_J = squash(s_J)
    for r_iter in range(iter_routing):

        gamma_IJ = softmax(delta_IJ, axis=2)

        if r_iter == iter_routing - 1:
            s_J = torch.mul(gamma_IJ, u_hat)
            s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
            v_J = squash(s_J)
        elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
            s_J = torch.mul(gamma_IJ, u_hat_stopped)
            s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
            v_J = squash(s_J)
            v_J_tiled = v_J.repeat([1, input_shape[1], 1, 1, 1])
            u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
            delta_IJ =delta_IJ+ u_produce_v

    return (v_J)


class Treecaps(nn.Module):
    def __init__(self, n_vol, feature_size, label_size, conv_feature, cov_size, num_cov, train_num_feats, num_dims, num_outputs, tlen, Wemd):
        super(Treecaps, self).__init__()

        self.feature_size = feature_size
        self.label_size = label_size
        self.conv_feature = conv_feature
        self.num_cov = num_cov
        self.cov_size = cov_size
        self.w_t = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature),device=cuda1),
                                std=1.0 / math.sqrt(train_num_feats)))
        self.w_l = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature),device=cuda1),
                                std=1.0 / math.sqrt(train_num_feats)))
        self.w_r = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature),device=cuda1),
                                std=1.0 / math.sqrt(train_num_feats)))
        self.init = truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats))
        self.b_conv = torch.nn.Parameter(torch.as_tensor(self.init, device=cuda1))
        self.w_t_1 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_l_1 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_r_1 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.b_conv_1 = torch.nn.Parameter(
            torch.as_tensor(truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats)),
                            device=cuda1))
        self.w_t_2 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_l_2 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_r_2 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.b_conv_2 = torch.nn.Parameter(
            torch.as_tensor(truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats)),
                            device=cuda1))
        self.w_t_3 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_l_3 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_r_3 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.b_conv_3 = torch.nn.Parameter(
            torch.as_tensor(truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats)),
                            device=cuda1))
        self.w_t_4 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_l_4 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_r_4 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.b_conv_4 = torch.nn.Parameter(
            torch.as_tensor(truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats)),
                            device=cuda1))
        self.w_t_5 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_l_5 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_r_5 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.b_conv_5 = torch.nn.Parameter(
            torch.as_tensor(truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats)),
                            device=cuda1))
        self.w_t_6 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_l_6 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_r_6 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.b_conv_6 = torch.nn.Parameter(
            torch.as_tensor(truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats)),
                            device=cuda1))

        self.w_t_7 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_l_7 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.w_r_7 = torch.nn.Parameter(truncated_normal_(torch.zeros((train_num_feats, conv_feature), device=cuda1),
                                                          std=1.0 / math.sqrt(train_num_feats)))
        self.b_conv_7 = torch.nn.Parameter(
            torch.as_tensor(truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats)),
                            device=cuda1))

        self.dr_w = torch.nn.Parameter(torch.nn.init.normal(torch.zeros((1, 576, num_dims * num_outputs, 8, 1), dtype=torch.float32, device=cuda1),
                                    std=0.01))
        self.dr_b = torch.nn.Parameter(torch.randn((1, 1, num_outputs, num_dims, 1), dtype=torch.float32,device=cuda1))
        self.Wemd = Wemd
        self.embed = torch.nn.Embedding(n_vol, feature_size)
        self.embed.weight.data.copy_(self.Wemd)

        self.top_a = 9
        self.top_b = 9
        self.caps1_num_dims = 8
        self.caps1_num_caps = int(num_cov * conv_feature // self.caps1_num_dims) * self.top_a
        self.caps1_out_caps = 1 #label size
        self.caps1_out_dims = 16

        self.hidden2label = nn.Linear(self.caps1_out_dims, 1)

    def hidden_layer(self,pooled):

        return torch.tanh(torch.matmul(pooled, self.w_h) + self.b_h)

    def encode(self, nodes, children):
        nodes = self.embed(nodes)

        conv = [
            torch.unsqueeze(conv_step(nodes, children, self.feature_size, self.w_t, self.w_l, self.w_r, self.b_conv),
                            dim=-1),
            torch.unsqueeze(
                conv_step(nodes, children, self.feature_size, self.w_t_1, self.w_l_1, self.w_r_1, self.b_conv_1),
                dim=-1),
            torch.unsqueeze(
                conv_step(nodes, children, self.feature_size, self.w_t_2, self.w_l_2, self.w_r_2, self.b_conv_2),
                dim=-1),
            torch.unsqueeze(
                conv_step(nodes, children, self.feature_size, self.w_t_3, self.w_l_3, self.w_r_3, self.b_conv_3),
                dim=-1),
            torch.unsqueeze(
                conv_step(nodes, children, self.feature_size, self.w_t_4, self.w_l_4, self.w_r_4, self.b_conv_4),
                dim=-1),
            torch.unsqueeze(
                conv_step(nodes, children, self.feature_size, self.w_t_5, self.w_l_5, self.w_r_5, self.b_conv_5),
                dim=-1),
            torch.unsqueeze(
                conv_step(nodes, children, self.feature_size, self.w_t_6, self.w_l_6, self.w_r_6, self.b_conv_6),
                dim=-1),
            torch.unsqueeze(
                conv_step(nodes, children, self.feature_size, self.w_t_7, self.w_l_7, self.w_r_7, self.b_conv_7),
                dim=-1)
        ]
        conv_output = torch.cat(conv, axis=-1)
        pri_capsules = torch.reshape(conv_output, shape=(1, -1, self.cov_size, self.num_cov))
        primary_static_caps = vts_routing(pri_capsules, self.top_a, self.top_b, self.caps1_num_caps,
                                          self.caps1_num_dims)
        primary_static_caps = torch.reshape(primary_static_caps, shape=(1, -1, 1, self.caps1_num_dims, 1))
        dr_shape = [1, self.caps1_num_caps, 1, self.caps1_num_dims, 1]
        codeCaps = dynamic_routing(dr_shape, primary_static_caps, self.dr_w, self.dr_b, num_outputs=self.caps1_out_caps,
                                   num_dims=self.caps1_out_dims)
        codeCaps = torch.squeeze(codeCaps, axis=1)


        return codeCaps

    def forward(self, nodes1,children1,nodes2,children2):

        lvec, rvec = self.encode(nodes1,children1), self.encode(nodes2,children2)

        abs_dist = torch.abs(torch.add(lvec, -rvec))
        abs_dist = torch.squeeze(torch.squeeze(abs_dist,dim=0),dim=2)
        y = torch.sigmoid(self.hidden2label(abs_dist))

        return y







