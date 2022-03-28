import torch
import torch.nn as nn
import numpy as np
import math

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


class transformer_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_len, pretrained_weight=None):
        super(transformer_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pad_len = pad_len
        self.pos_emb = PositionalEncoding(0.2, embedding_dim, 500)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_encode = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.drop = nn.Dropout(0.2)
        self.fc_1 = nn.Linear(embedding_dim, embedding_dim)
        self.activate = nn.Tanh()

        #self.dropout = nn.Dropout(dropout)
        if pretrained_weight is not None:
            pad_np = np.zeros((1, embedding_dim))
            bos_np = np.ones((1, embedding_dim))
            pretrained_weight = np.concatenate((pad_np, bos_np, pretrained_weight), axis=0)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, x):
        max_len = self.pad_len
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
        out = self.transformer_encode(emb, src_key_padding_mask=mask)

        out = self.fc_1(out[0, :, :])

        return out

class lstm_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_len, pretrained_weight=None):
        super(lstm_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pad_len = pad_len
        self.lstm = nn.LSTM(embedding_dim, 100, bidirectional=True, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(200, 200)
        if pretrained_weight is not None:
            pad_np = np.zeros((1, embedding_dim))
            pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, x):
        bs = len(x)
        max_len = self.pad_len
        seq = []
        for string in x:
            if len(string) > max_len:
                seq.append(string[0:max_len])
            else:
                seq.append(np.pad(np.array(string), (0, max_len - len(string)), 'constant', constant_values=-1).tolist())
        encodes = torch.LongTensor(seq).cuda()
        emb = self.embedding(encodes+1)
        leng=[]
        for i in x:
            if len(i) < max_len:
                leng.append(len(i))
            else:
                leng.append(max_len)
        emb_packed = nn.utils.rnn.pack_padded_sequence(emb, leng, batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(emb_packed)
        out = self.fc(hn.transpose(0,1).reshape(bs,-1))

        return out


class CloneDetector(nn.Module):
    def __init__(self, MAX_TOKENS, EMBEDDING_DIM, pad_len, model, pretrained_weight=None):
        super(CloneDetector, self).__init__()
        if model == 'transformer':
            self.encoder = transformer_encoder(MAX_TOKENS+2, EMBEDDING_DIM, pad_len, pretrained_weight)
        elif model == 'lstm':
            self.encoder = lstm_encoder(MAX_TOKENS, EMBEDDING_DIM, pad_len, pretrained_weight)
        self.hidden2label = nn.Linear(EMBEDDING_DIM, 1)

    def forward(self, x1, x2):
        lvec = self.encoder(x1)
        rvec = self.encoder(x2)

        abs_dist = torch.abs(torch.add(lvec, -rvec))
        y = torch.sigmoid(self.hidden2label(abs_dist))

        return y