import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing,GatedGraphConv
from  .nn.glob import GlobalAttention
import torch.nn.functional as F
from torch import einsum
import numpy as np

class RankLoss(nn.Module):
    def __init__(self, margin):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, nl_vec_pos, nl_vec_neg, code_vec_pos, code_vec_neg):
        return (self.margin - F.cosine_similarity(nl_vec_pos,
                                                  code_vec_pos) + F.cosine_similarity(
            nl_vec_neg, code_vec_pos)).clamp(min=1e-6).mean()

class GGNN(torch.nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device,token_embedding):
        super(GGNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embed = token_embedding
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
        node_text = self.embed(node_text+1)
        # node_type = self.type_embed(node_type)
        # node = self.embed_fusion(torch.cat([node_text, node_type], dim=-1))
        edge_weight = self.edge_embed(edge_attr-1)
        edge_weight = edge_weight.mean(1)
        x = self.ggnnlayer(node_text, edge_index, edge_weight)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        hg = self.pool(x, batch=batch).squeeze(0)
        # out = self.out(hg)

        return hg.view(-1, self.embedding_dim)

class DocumentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, token_embedding):
        super(DocumentEncoder, self).__init__()
        self.embedding = token_embedding
        self.embedding_dim = embedding_dim

        self.W_b = nn.Parameter(torch.rand((embedding_dim, embedding_dim), dtype=torch.float, requires_grad=True)).cuda()

        # self.use_gpu = use_gpu
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
        # for docstring in x:
        if len(x) > max_len:
            seq.append(x[0: max_len])
        else:
            seq.append(np.pad(np.array(x), (0, max_len - len(x)), 'constant', constant_values=-1).tolist())
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
    def __init__(self, vocablen, embedding_dim, num_layers, device, pretrained_weight=None):
        super(SearchModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocablen + 1   # +1 because pad
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.encoder = GGNN(vocablen, embedding_dim, num_layers, device, self.token_embedding)
        self.encoder_doc = DocumentEncoder(vocablen, embedding_dim, self.token_embedding)
        if pretrained_weight is not None:
            pad_np = np.zeros((1, self.embedding_dim))
            # bos_np = np.ones((1, self.embedding_dim))
            pretrained_weight = np.concatenate((pad_np, pretrained_weight), axis=0)
            self.token_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, x1, x2):
        lvec = self.encoder(x1)
        rvec = self.encoder_doc(x2, 20)

        return lvec, rvec