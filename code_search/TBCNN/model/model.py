import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch import einsum
import math
from model.transformer_encoder import transformer_encoder
from model.lstm_encoder import lstm_encoder
from model.tbcnn_encoder import tbcnn_encoder


class DocumentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, use_gpu, pretrained_weight=None):
        super(DocumentEncoder, self).__init__()
        # embedding_dim = 200
        self.embedding = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim

        self.W_b = nn.Parameter(torch.rand((embedding_dim, embedding_dim), dtype=torch.float, requires_grad=True)).cuda()

        self.use_gpu = use_gpu
        self.max_index = vocab_size

        # pretrained  embedding
        if pretrained_weight is not None:
            pad_np = np.zeros((1, self.embedding_dim))
            # bos_np = np.ones((1, self.embedding_dim))
            pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

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
        batch_size = len(x)
        seq = []
        for docstring in x:
            # docstring = (docstring.replace("\n", "")).split(' ')
            if len(docstring) > max_len:
                seq.append(docstring[0: max_len])
            else:
                seq.append(
                    np.pad(np.array(docstring), (0, max_len - len(docstring)), 'constant', constant_values=-1).tolist())
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


class SearchModel(nn.Module):
    def __init__(self, code_model,embedding_dim,vocab_size,feature_size,conv_feature, w_t, w_l, w_r, b_conv, w_h, b_h, use_gpu=True, pretrained_weight=None):
        super(SearchModel, self).__init__()
        self.gpu = use_gpu
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.feature_size = feature_size
        # self.label_size = label_size
        self.conv_feature = conv_feature

        self.w_t = w_t
        self.w_l = w_l
        self.w_r = w_r
        self.b_conv = b_conv
        self.w_h = w_h
        self.b_h = b_h

        self.encoder_doc = DocumentEncoder(self.vocab_size, self.embedding_dim, self.gpu, pretrained_weight)
        # if code_model == "LSTM":
        #     self.encode = lstm_encoder(self.vocab_size, self.embedding_dim, pretrained_weight)
        # elif code_model == "Transformer":
        #     self.encode = transformer_encoder(self.vocab_size, self.embedding_dim, pretrained_weight)
        self.encoder = tbcnn_encoder(self.feature_size * 2,self.conv_feature,self.w_t, self.w_l, self.w_r, self.b_conv, self.w_h, self.b_h)

    def forward(self, x1, x2,x3,p1):

        lvec = self.encoder(x1,x2,p1)
        rvec = self.encoder_doc(x3, 12)

        return lvec, rvec