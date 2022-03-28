import torch
import torch.nn as nn
import torch.nn.functional as F


class Code2VecEncoder(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, code_vector_size, dropout):
        super(Code2VecEncoder, self).__init__()

        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(3 * embedding_dim, code_vector_size, bias=False)
        a = torch.nn.init.uniform_(torch.empty(code_vector_size, 1, dtype=torch.float32, requires_grad=True))
        self.a = nn.parameter.Parameter(a, requires_grad=True)

    def forward(self, starts, paths, ends, masks):
        batch_size = starts.shape[0]
        starts_embedded = self.node_embedding(starts)
        paths_embedded = self.path_embedding(paths)
        ends_embedded = self.node_embedding(ends)

        # Concatenate
        context_embedded = torch.cat((starts_embedded, paths_embedded, ends_embedded), dim=2)
        context_embedded = self.dropout(context_embedded)

        # Fully Connected
        context_after_dense = torch.tanh(self.fc(context_embedded))

        # Attention weight
        attention_weight = self.a.repeat(batch_size, 1, 1)  # attention_weight = [bs, code_vector_size, 1]
        attention_weight = torch.bmm(context_after_dense, attention_weight)
        # masks = (torch.log(masks)).unsqueeze(2)   # masks = [bs, max_seq_length, 1]
        # attention_weight = attention_weight + masks
        attention_weight = F.softmax(attention_weight, dim=1)  # attention_weight = [bs, max_seq_length, 1]

        # code_vectors = [bs, code_vector_size]
        code_vectors = torch.sum(torch.mul(context_after_dense, attention_weight.expand_as(context_after_dense)), dim=1)

        return code_vectors


class Code2VecCloneDetector(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, code_vector_size, dropout):
        super(Code2VecCloneDetector, self).__init__()
        self.encoder = Code2VecEncoder(nodes_dim, paths_dim, embedding_dim, code_vector_size, dropout)
        self.hidden2label = nn.Linear(code_vector_size, 1)

    def forward(self, starts_x1, paths_x1, ends_x1, masks_x1, starts_x2, paths_x2, ends_x2, masks_x2):
        lvec = self.encoder(starts_x1, paths_x1, ends_x1, masks_x1)
        rvec = self.encoder(starts_x2, paths_x2, ends_x2, masks_x2)
   
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        y = torch.sigmoid(self.hidden2label(abs_dist))
        
        return y.view(masks_x1.size(0))













