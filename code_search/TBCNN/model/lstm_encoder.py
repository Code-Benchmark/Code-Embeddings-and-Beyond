import torch
import torch.nn as nn
import numpy as np


class lstm_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight):
        super(lstm_encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim*2,embedding_dim*2),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.fc = nn.Linear(embedding_dim*2, embedding_dim)
        if pretrained_weight is not None:
            pad_np = np.zeros((1, embedding_dim))
            pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, x):
        max_len = 500
        seq = []
        leng = []
        for docstring in x:
            if len(docstring) > max_len:
                seq.append(docstring[0 : max_len])
                leng.append(max_len)
            else:
                seq.append(np.pad(np.array(docstring), (0, max_len - len(docstring)), 'constant', constant_values=-1).tolist())
                leng.append(len(docstring))
        encodes = torch.LongTensor(seq).cuda()
        emb = self.embedding(encodes+1)
        emb_packed = nn.utils.rnn.pack_padded_sequence(emb, leng, batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(emb_packed)
        # emb_out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        hn = hn[-2:, :, :]
        hidden = hn.permute((1, 0, 2)).contiguous().view(-1,self.embedding_dim*2)
        out = self.fc(hidden)
        # out = torch.mean(emb_out, dim=1)

        return out