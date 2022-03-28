from torch import nn
# from .embedding import LeftEmbedding, RightEmbedding, PathEmbedding, QueryEmbedding
from model.embedding.tokens import LeftEmbedding, RightEmbedding, QueryEmbedding
from model.embedding.paths import PathEmbedding
from .encoder import Encoder
import torch
from torch import einsum
import math
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, s_vocab, t_vocab=None):
        super().__init__()
        self.args = args
        self.left_embedding = LeftEmbedding(args, s_vocab)
        self.query_embedding = QueryEmbedding(args,s_vocab)
        if args.relation_path or args.absolute_path:
            self.path_embedding = PathEmbedding(args)

        self.encoder = Encoder(args)
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.relation_path = args.relation_path
        self.absolute_path = args.absolute_path

        self.W_a = nn.Parameter(
                    torch.rand((args.hidden, args.hidden),
                            dtype=torch.float, requires_grad=True))
        nn.init.xavier_uniform_(self.W_a)
        self.linear = nn.Linear(args.hidden, args.embedding_size)
        self.W_b = nn.Parameter(
                    torch.rand((args.embedding_size, args.embedding_size),
                            dtype=torch.float, requires_grad=True))
                            
        nn.init.xavier_uniform_(self.W_b)


    def attention(self, encoder_output_bag, hidden, mask, w, type):
        e_out = torch.einsum('ble,ek->blk', encoder_output_bag, w)
        if type == 'code':
            at = F.softmax(torch.einsum('ble,bej->bl', e_out, hidden.permute(1, 2, 0)), dim=-1) \
                .masked_fill(mask, 1e-20)
        elif type == 'query':
            at = F.softmax(torch.einsum('ble,bej->bl', e_out, hidden.permute(1, 2, 0)), dim=-1)
        ct = torch.einsum('bi,bij->bj', at, encoder_output_bag)
        return ct

    def encode_query(self, query):
        # nl_emb = self.query_embedding(query)
        nl_emb = query
        # print(nl_emb.shape)
        mask = None
        # nl_emb_split = torch.split(nl_emb.view(-1, self.args.embedding_size),
        #                      self.args.max_doc_length, dim=0)
        hidden_0 = nl_emb.mean(1).unsqueeze(0)
        # print(hidden_0.shape)
        # print(self.W_b.shape)
        # hidden_0 = [ne.mean(0).unsqueeze(dim=0) for ne in nl_emb_split]
        # hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)
        doc_out = self.attention(nl_emb, hidden_0, mask, self.W_b,'query')

        return doc_out


    def encode(self, data):
        content = data['content']
        content_mask = data['content_mask']
        path_map = data['path_map']
        paths = data['paths']
        paths_mask = data['paths_mask']
        r_paths = data['r_paths']
        r_paths_mask = data['r_paths_mask']
        r_path_idx = data['r_path_idx']
        named = data['named']

        content_ = self.left_embedding(content, named)
        if self.relation_path:
            paths_ = self.path_embedding(paths, paths_mask, type='relation')
        else:
            paths_ = None
        if self.absolute_path:
            r_paths_ = self.path_embedding(r_paths, r_paths_mask, type='absolute')
        else:
            r_paths_ = None
        mask_ = (content_mask > 0).unsqueeze(1).repeat(1, content_mask.size(1), 1).unsqueeze(1)
        # bs, 1,max_code_length,max_code_length

        memory = self.encoder(content_, mask_, paths_, path_map, r_paths_, r_path_idx)
        # bs, max_code_length, hidden
        return memory, (content_mask == 0)


    def forward(self, data):
        memory, memory_key_padding_mask = self.encode(data)

        code_vec_atten = torch.mean(memory, dim=1)
        # code_vec_atten = memory[:,0,:]
        code_vec = self.linear(code_vec_atten)

        nl_vec = self.encode_query(data['query'])
        neg_nl_vec = self.encode_query(data['fp_query'])
        # cos = F.cosine_similarity(nl_vec, code_vec).view(nl_vec.size(0), -1)
        # t = torch.zeros_like(cos)
        # out = torch.cat([t, cos], dim=1)
        # return out

        return code_vec, nl_vec, neg_nl_vec


