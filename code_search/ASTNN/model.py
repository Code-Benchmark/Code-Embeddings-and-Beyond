import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch import einsum


class RankLoss(nn.Module):
    def __init__(self, margin):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, nl_vec_pos, nl_vec_neg, code_vec_pos, code_vec_neg):
        return (self.margin - F.cosine_similarity(nl_vec_pos,
                                                  code_vec_pos) + F.cosine_similarity(
            nl_vec_neg, code_vec_pos)).clamp(min=1e-6).mean()


class BatchTreeEncoder(nn.Module):
    def __init__(self, token_embedding,vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = token_embedding
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.max_index = vocab_size
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.embedding_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] is not -1:
            index.append(i)
            current_node.append(node[i][0])
            temp = node[i][1:]
            c_num = len(temp)
            for j in range(c_num):
                if temp[j][0] is not -1:
                    if len(children_index) <= j:
                        children_index.append([i])
                        children.append([temp[j]])
                    else:
                        children_index[j].append(i)
                        children[j].append(temp[j])
        # else:
        #     batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor((current_node))+1))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class DocumentEncoder(nn.Module):
    def __init__(self, token_embedding, vocab_size, embedding_dim, use_gpu, pretrained_weight=None):
        super(DocumentEncoder, self).__init__()
        # embedding_dim = 200
        # self.embedding = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.embedding = token_embedding
        self.embedding_dim = embedding_dim
        self.W_b = nn.Parameter(torch.rand((embedding_dim, embedding_dim), dtype=torch.float, requires_grad=True)).cuda()
        self.use_gpu = use_gpu
        self.max_index = vocab_size
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_b)

    def attention(self, encoder_output_bag, hidden, lengths_k, w):
        e_out = torch.cat(encoder_output_bag, dim=0)
        ha = einsum('ij,jk->ik', e_out, w)
        ha = torch.split(ha, lengths_k, dim=0)
        hd = hidden.transpose(0, 1)
        hd = torch.unbind(hd, dim=0)
        at = [F.softmax(torch.einsum('ij,kj->i', _ha, _hd), dim=0) for
              _ha, _hd in zip(ha, hd)]
        ct = [torch.einsum('i,ij->j', a, e).unsqueeze(0) for a, e in
              zip(at, encoder_output_bag)]
        ct = torch.cat(ct, dim=0).unsqueeze(0)

        return ct

    def forward(self, x, max_len):  # , encoder_out, encoder_lens):k
        seq = []
        for docstring in x:
            if len(docstring) > max_len:
                seq.append(docstring[0: max_len])
            else:
                seq.append(np.pad(np.array(docstring), (0, max_len - len(docstring)), 'constant', constant_values=-1).tolist())
        encodes = torch.LongTensor(seq).cuda()
        nl_emb = self.embedding(encodes + 1)
        nl_emb = torch.split(nl_emb.view(-1, self.embedding_dim),
                             max_len, dim=0)
        hidden_0 = [ne.mean(0).unsqueeze(dim=0) for ne in nl_emb]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)
        nl_emb_atten = self.attention(nl_emb, hidden_0,
                                      max_len, self.W_b)
        doc_out = nl_emb_atten[-1]

        return doc_out


class ASTNN4Search(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, batch_size, use_gpu=True, pretrained_weight=None):
        super(ASTNN4Search, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size + 1   # +1 because pad
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.encoder = BatchTreeEncoder(self.token_embedding, self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu)
        self.encoder_doc = DocumentEncoder(self.token_embedding, self.vocab_size, self.embedding_dim, self.gpu)

        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.linear = nn.Linear(self.hidden_dim * 2, 128)

        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.5)

        if pretrained_weight is not None:
            pad_np = np.zeros((1, self.embedding_dim))
            # bos_np = np.ones((1, self.embedding_dim))
            pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
            self.token_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # self.token_embedding.weight.requires_grad = False

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda()
                c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda()
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def encode(self, x, x2):
        batch_size = len(x2)
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(batch_size):
            end += lens[i]
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(batch_size, max_len, -1)
        # return encodes

        gru_out, hidden = self.bigru(encodes, self.hidden)
        gru_out = torch.transpose(gru_out, 1, 2)

        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        return self.linear(gru_out)#, lens

    def forward(self, x1, x2):
        lvec = self.encode(x1,x2)
        rvec = self.encoder_doc(x2, 20)
        # cos = F.cosine_similarity(rvec,lvec).view(1,-1)
        # return cos
        return lvec, rvec  # scores, loss
