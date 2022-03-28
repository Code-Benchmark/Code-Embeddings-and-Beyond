import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing,GatedGraphConv
from torch_geometric.nn.glob import GlobalAttention

class GGNN(torch.nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device):
        super(GGNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.type_dim = 47
        self.embed = nn.Embedding(vocablen, embedding_dim)
        # self.type_embed = nn.Embedding(self.type_dim, embedding_dim)
        # self.embed_fusion = nn.Linear(embedding_dim*2, embedding_dim)
        self.edge_embed = nn.Embedding(7, embedding_dim)
        self.ggnnlayer = GatedGraphConv(embedding_dim,num_layers)
        self.mlp_gate = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)
        # self.out = nn.Linear(embedding_dim, 104)

    def forward(self, data):
        node_text = torch.LongTensor(data['node_ids']).cuda()
        # node_type = torch.LongTensor(data['node_types']).cuda()
        edge_index = torch.LongTensor(data['edges']).cuda()
        edge_attr = torch.LongTensor(data['edge_types']).cuda()
        node_text = self.embed(node_text)
        edge_weight = self.edge_embed(edge_attr-1).mean(1)
        x = self.ggnnlayer(node_text, edge_index, edge_weight)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        hg = self.pool(x, batch=batch).squeeze(0)
        return hg


class GGNNCloneDetector(nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device):
        super(GGNNCloneDetector, self).__init__()
        self.encoder = GGNN(vocablen, embedding_dim, num_layers, device)
        self.hidden2label = nn.Linear(embedding_dim, 1)

    def forward(self, data1, data2):
        lvec = self.encoder(data1)
        rvec = self.encoder(data2)

        abs_dist = torch.abs(torch.add(lvec, -rvec))
        y = torch.sigmoid(self.hidden2label(abs_dist))

        return y