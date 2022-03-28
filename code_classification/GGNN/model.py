import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing,GatedGraphConv
from torch_geometric.nn.glob import GlobalAttention

class GGNN(torch.nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device):
        super(GGNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocablen, embedding_dim)
        self.edge_embed = nn.Embedding(7, embedding_dim)
        self.ggnnlayer = GatedGraphConv(embedding_dim,num_layers)
        self.mlp_gate = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)
        self.out = nn.Linear(embedding_dim, 104)

    def forward(self, data):
        x = torch.LongTensor(data['node_ids']).cuda()
        edge_index = torch.LongTensor(data['edges']).cuda()
        edge_attr = torch.LongTensor(data['edge_types']).cuda()
        x = self.embed(x)
        edge_weight = self.edge_embed(edge_attr-1).mean(1)
        x = self.ggnnlayer(x, edge_index, edge_weight)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        hg = self.pool(x, batch=batch).squeeze(0)
        out = self.out(hg)

        return out.unsqueeze(0)