from torch import nn
from .embedding import LeftEmbedding, RightEmbedding, PathEmbedding
from .encoder import Encoder
import torch
from torch import einsum
import math
import torch.nn.functional as F


class Model(nn.Module):
    # def __init__(self, args, s_vocab, t_vocab):
    def __init__(self, args, s_vocab):
        super().__init__()
        self.args = args
        self.left_embedding = LeftEmbedding(args, s_vocab)
        # self.right_embedding = RightEmbedding(args, t_vocab)
        if args.relation_path or args.absolute_path:
            self.path_embedding = PathEmbedding(args)
        self.encoder = Encoder(args)
        self.hidden_size = args.hidden
        self.softmax = nn.LogSoftmax(dim=-1)
        self.relation_path = args.relation_path
        self.absolute_path = args.absolute_path
        # self.content_mask = None
        # self.path_map = None
        # self.r_path_idx = None
        # if self.args.uni_vocab:
        #     self.right_embedding.embedding.weight = self.left_embedding.embedding.weight
        #     if args.embedding_size != args.hidden:
        #         self.right_embedding.in_.weight = self.left_embedding.in_.weight
        # if self.args.weight_tying:
        #     self.right_embedding.out.weight = self.right_embedding.embedding.weight

        self.W_a = nn.Parameter(
            torch.rand((self.hidden_size, self.hidden_size),
                       dtype=torch.float, requires_grad=True))
        self.hidden2label = nn.Linear(args.embedding_size, 104)
        self.fc = nn.Linear(self.hidden_size, args.embedding_size)

    def attention(self, encoder_output_bag, hidden, mask, w):
        e_out = torch.einsum('ble,ek->blk', encoder_output_bag, w)
        at = F.softmax(torch.einsum('ble,bej->bl', e_out, hidden.permute(1, 2, 0)), dim=-1) \
            .masked_fill(mask, 1e-20)
        ct = torch.einsum('bi,bij->bj', at, encoder_output_bag)
        return ct

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

    def prepare_attr(self, content_mask, path_map, r_path_idx):
        self.content_mask = content_mask
        self.path_map = path_map
        self.r_path_idx = r_path_idx
        # named = data['named']

    def encode_attr(self, content_, paths_, r_paths_):

        # content_ = self.left_embedding(content, named)
        content_ = content_
        if self.relation_path:
            # paths_ = self.path_embedding(paths, paths_mask, type='relation')
            paths_ = paths_#.permute(0,2,1)
        else:
            paths_ = None
        if self.absolute_path:
            # r_paths_ = self.path_embedding(r_paths, r_paths_mask, type='absolute')
            r_paths_ = r_paths_#.permute(0,2,1)
        else:
            r_paths_ = None

        # bs = content_.size(0)
        # if bs != self.content_mask.size(0):
        #     self.content_mask = self.content_mask.repeat(bs, 1)
        #     self.path_map = self.path_map.repeat(bs, 1, 1)
        #     self.r_path_idx = self.r_path_idx.repeat(bs, 1)


        mask_ = (self.content_mask > 0).unsqueeze(1).repeat(1, self.content_mask.size(1), 1).unsqueeze(1)
        # bs, 1,max_code_length,max_code_length

        memory = self.encoder(content_, mask_, paths_, self.path_map, r_paths_, self.r_path_idx)
        # bs, max_code_length, hidden
        return memory, (self.content_mask == 0)

    # def forward(self, data):
    #     memory, memory_key_padding_mask = self.encode(data)
    def forward(self, content_, paths_, r_paths_):
        memory, memory_key_padding_mask = self.encode_attr(content_, paths_, r_paths_)
        # ATTENTION
        # hidden_0 = memory.mean(1).unsqueeze(0)
        # code_vec_atten = self.attention(memory, hidden_0, memory_key_padding_mask, self.W_a.cuda())
        # return memory
        # MEAN
        code_vec_atten = torch.mean(memory, dim=1)
        # #
        # # EOS
        # code_vec_atten = memory[:,0,:]

        code_vectors = self.fc(code_vec_atten)
        out = self.hidden2label(code_vectors)
        return F.softmax(out,dim=-1)


