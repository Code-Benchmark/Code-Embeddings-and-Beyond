import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class lstm_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight=None):
        super(lstm_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 200, bidirectional=True, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(200*2, 104)
        if pretrained_weight is not None:
            pad_np = np.zeros((1, embedding_dim))
            pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, x):
        bs = x.size(0)
        # emb = self.embedding(x)
        # emb = self.drop(x)
        emb = x
        _, (hn, cn) = self.lstm(emb)
        hn = hn[-2:, :, :]
        out = self.fc(hn.permute((1, 0, 2)).contiguous().view(bs, -1))
        out = F.softmax(out, dim=-1)
        return out