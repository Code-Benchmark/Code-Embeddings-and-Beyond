import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch import einsum
import math
from transformer_encoder import transformer_encoder
from lstm_encoder import lstm_encoder


class DocumentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, use_gpu, embedding_layer, pretrained_weight=None):
        super(DocumentEncoder, self).__init__()
        # embedding_dim = 200
        # self.embedding = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.embedding = embedding_layer
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
        # batch_size = len(x)
        seq = []
        for docstring in x:
            if len(docstring) > max_len:
                seq.append(docstring[0: max_len])
            else:
                seq.append(np.pad(np.array(docstring), (0, max_len - len(docstring)), 'constant', constant_values=-2).tolist())
        encodes = torch.LongTensor(seq).cuda()
        nl_emb = self.embedding(encodes + 2)   # transformer +2 lstm +1
        # nl_emb = x
        nl_emb = torch.split(nl_emb.view(-1, self.embedding_dim),
                             max_len, dim=0)
        hidden_0 = [ne.mean(0).unsqueeze(dim=0) for ne in nl_emb]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)
        nl_emb_atten = self.attention(nl_emb, hidden_0,
                                      max_len, self.W_b)
        doc_out = nl_emb_atten[-1]

        return doc_out


class SearchModel(nn.Module):
    def __init__(self, code_model, embedding_dim, vocab_size, use_gpu=True, pretrained_weight=None):
        super(SearchModel, self).__init__()
        self.gpu = use_gpu
        self.vocab_size = vocab_size + 2   # transformer +2 lstm +1
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.encoder_doc = DocumentEncoder(self.vocab_size, self.embedding_dim, self.gpu, self.embedding_layer)
        if code_model == "LSTM":
            self.encode = lstm_encoder(self.vocab_size, self.embedding_dim, self.embedding_layer)
            if pretrained_weight is not None:
                pad_np = np.zeros((1, self.embedding_dim))
                # bos_np = np.ones((1, self.embedding_dim))
                pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
                self.embedding_layer.weight.data.copy_(torch.from_numpy(pretrained_weight))
        elif code_model == "Transformer":
            self.encode = transformer_encoder(self.vocab_size, self.embedding_dim, self.embedding_layer)
            if pretrained_weight is not None:
                pad_np = np.zeros((1, embedding_dim))
                bos_np = np.ones((1, embedding_dim))
                pretrained_weight = np.concatenate((pad_np, bos_np, pretrained_weight), axis=0)
                self.embedding_layer.weight.data.copy_(torch.from_numpy(pretrained_weight))


        # self.embedding_layer.weight.requires_grad = False

    def forward(self, x1, x2):
        lvec = self.encode(x1)
        rvec = self.encoder_doc(x2, 20)

        # cos = F.cosine_similarity(rvec,lvec).view(1,-1)
        # return cos
        return lvec, rvec