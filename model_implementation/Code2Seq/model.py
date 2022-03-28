import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.nn.init as weight_init


class Code2Seq(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, code_vector_size, output_dim, dropout):
        super(Code2Seq, self).__init__()
        self.max_path_num = 8
        self.max_contexts = 200
        self.embedding_dim = embedding_dim
        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim, padding_idx=0)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(code_vector_size, code_vector_size)
        self.path_lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True,
                                 bidirectional=True)
        self.W_a = nn.Parameter(
            torch.rand((embedding_dim * 4, embedding_dim * 4),
                       dtype=torch.float, requires_grad=True))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_a)
        nn.init.uniform_(self.path_embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.path_embedding.weight[0], 0)
        for w in self.path_lstm.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

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

    def forward(self, starts, paths, ends):
        batch_size = starts.size(0)

        starts_embedded = torch.sum(self.node_embedding(starts), dim=2)
        ends_embedded = torch.sum(self.node_embedding(ends), dim=2)
        #
        paths_embedded = self.path_embedding(paths)
        #
        p = paths_embedded.view(batch_size * self.max_contexts, self.max_path_num, -1)
        out, (hn, cn) = self.path_lstm(p)
        hn = hn[-2:, :, :]
        hidden = hn.permute((1, 0, 2)).contiguous().view(batch_size * self.max_contexts, 1, -1)
        path_vec = hidden.squeeze(1).view(batch_size, self.max_contexts, -1)

        # Concatenate
        context_embedded = torch.cat((starts_embedded, path_vec, ends_embedded), dim=2)
        context_embedded = self.dropout(context_embedded)

        # Attention
        code_vec = torch.split(
            context_embedded.contiguous().view(-1, 4 * self.embedding_dim),
            self.max_contexts, dim=0)
        hidden_0 = [cv.mean(0).unsqueeze(dim=0) for cv in code_vec]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)
        code_vec_atten = self.attention(code_vec, hidden_0, self.max_contexts,
                                        self.W_a)
        code_vectors = self.fc(code_vec_atten[-1])
        
        return code_vectors