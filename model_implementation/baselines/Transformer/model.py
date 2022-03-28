import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=200):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight=None):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size+2, embedding_dim,padding_idx=0)
        self.pos_emb = PositionalEncoding(0.2, embedding_dim, 500)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_encode = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

        if pretrained_weight is not None:
            pad_np = np.zeros((1, embedding_dim))
            bos_np = np.ones((1, embedding_dim))
            pretrained_weight = np.concatenate((pad_np, bos_np, pretrained_weight), axis=0)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, x, max_len):
        seq = []
        for string in x:
            if len(string) > max_len:
                seq.append(string[0:max_len])
            else:
                seq.append(np.pad(np.array(string), (0, max_len - len(string)), 'constant', constant_values=-2).tolist())
        
        encodes = torch.LongTensor(seq)
        inputs = torch.cat([torch.ones([encodes.size(0)], dtype=encodes.dtype).unsqueeze(1), encodes + 2], dim=1).cuda()
        mask = (inputs == 0)
        emb = self.embedding(inputs)
        emb = self.pos_emb(emb).transpose(0, 1).contiguous()
        code_vector = self.transformer_encode(emb, src_key_padding_mask=mask)
        code_vector = self.fc(out[0, :, :])

        return code_vector