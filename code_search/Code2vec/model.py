import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as weight_init
from torch import einsum


class RankLoss(nn.Module):
    def __init__(self, margin):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, nl_vec_pos, nl_vec_neg, code_vec_pos, code_vec_neg):
        return (self.margin - F.cosine_similarity(nl_vec_pos,
                                                  code_vec_pos) + F.cosine_similarity(
            nl_vec_neg, code_vec_pos)).clamp(min=1e-6).mean()


class Code2VecEncoder(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, code_vector_size, dropout, token_embedding=None):
        super(Code2VecEncoder, self).__init__()
        self.max_path_num = 8
        self.max_contexts = 500
        self.embedding_dim = embedding_dim
        self.code_vector_size = code_vector_size
        # self.node_embedding = nn.Embedding(nodes_dim+1, embedding_dim, padding_idx=0)
        self.node_embedding = token_embedding
        self.path_embedding = nn.Embedding(paths_dim+1, embedding_dim, padding_idx=0)
        # self.node_embedding = token_embedding
        # self.path_embedding = nn.Embedding(vocab_path_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(code_vector_size, code_vector_size, bias=False)
        a = torch.nn.init.uniform_(torch.empty(code_vector_size, 1, dtype=torch.float32, requires_grad=True))
        self.a = nn.parameter.Parameter(a, requires_grad=True)
        self.fc_2 = nn.Linear(code_vector_size, embedding_dim)

    def forward(self, starts, paths, ends, masks):
        batch_size = starts.size(0)
        starts_embedded = self.node_embedding(starts+1)
        paths_embedded = self.path_embedding(paths)
        ends_embedded = self.node_embedding(ends+1)

        # Concatenate
        context_embedded = torch.cat((starts_embedded, paths_embedded, ends_embedded), dim=2)
        context_embedded = self.dropout(context_embedded)
        # Fully Connected
        context_after_dense = torch.tanh(self.fc(context_embedded))

        # Attention weight
        attention_weight = self.a.repeat(batch_size, 1, 1)  # attention_weight = [bs, code_vector_size, 1]
        attention_weight = torch.bmm(context_after_dense, attention_weight)
        # masks = (torch.log(masks)).unsqueeze(2)  # masks = [bs, max_seq_length, 1]
        # attention_weight = attention_weight + masks
        attention_weight = F.softmax(attention_weight, dim=1)  # attention_weight = [bs, max_seq_length, 1]
        #
        # # code_vectors = [bs, code_vector_size]
        code_vectors = torch.sum(torch.mul(context_after_dense, attention_weight.expand_as(context_after_dense)), dim=1)

        # code_vectors = [bs, max_seq_length, code_vector_size]
        # code_vectors = torch.mul(context_after_dense, attention_weight.expand_as(context_after_dense))
        code_vectors = self.fc_2(code_vectors)

        return code_vectors

class DocumentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, token_embedded=None):
        super(DocumentEncoder, self).__init__()
        # embedding_dim = 200
        # self.embedding = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.embedding = token_embedded
        self.embedding_dim = embedding_dim
        self.W_b = nn.Parameter(torch.rand((embedding_dim, embedding_dim), dtype=torch.float, requires_grad=True)).cuda()

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
                seq.append(
                    np.pad(np.array(docstring), (0, max_len - len(docstring)), 'constant', constant_values=-1).tolist())

        encodes = torch.LongTensor(seq).cuda()
        nl_emb = self.embedding(encodes + 1)
        # nl_emb = x
        nl_emb = torch.split(nl_emb.view(-1, self.embedding_dim),
                             max_len, dim=0)
        hidden_0 = [ne.mean(0).unsqueeze(dim=0) for ne in nl_emb]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)
        nl_emb_atten = self.attention(nl_emb, hidden_0,
                                      max_len, self.W_b)
        doc_out = nl_emb_atten[-1]

        return doc_out

class CodeSearchModel(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, code_vector_size, vocab_size, pretrained_weight=None):
        super(CodeSearchModel, self).__init__()
        self.vocab_size = vocab_size + 1
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.encoder = Code2VecEncoder(nodes_dim, paths_dim, embedding_dim, code_vector_size, 0.25, self.token_embedding)
        self.encoder_doc = DocumentEncoder(vocab_size, embedding_dim, self.token_embedding)
        # self.nodes = nodes
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.token_embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.token_embedding.weight[0], 0)
        # if pretrained_weight is not None:
        #     pad_np = np.zeros((1, self.embedding_dim))
        #     # bos_np = np.ones((1, self.embedding_dim))
        #     pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
        #     self.token_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # self.token_embedding.weight.requires_grad = False

    def forward(self, starts, paths, ends, masks, x2):
    # def forward(self, x1, x2):
        lvec = self.encoder(starts, paths, ends, masks)
        rvec = self.encoder_doc(x2, 20)

        # cos = F.cosine_similarity(rvec,lvec).view(1,-1)
        # return cos
        return lvec, rvec
