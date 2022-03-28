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
        #self.right_embedding = RightEmbedding(args, t_vocab)
        if args.relation_path or args.absolute_path:
            self.path_embedding = PathEmbedding(args)
        self.encoder = Encoder(args)
        self.hidden_size = args.hidden
        self.softmax = nn.LogSoftmax(dim=-1)
        self.relation_path = args.relation_path
        self.absolute_path = args.absolute_path
        self.hidden2label = nn.Linear(args.embedding_size, 1)
        self.fc = nn.Linear(args.hidden, args.embedding_size)
    
    
    
    def encode(self, data, type):
        if type == 1:
            content = data['content']
            content_mask = data['content_mask']
            path_map = data['path_map']
            paths = data['paths']
            paths_mask = data['paths_mask']
            r_paths = data['r_paths']
            r_paths_mask = data['r_paths_mask']
            r_path_idx = data['r_path_idx']
            named = data['named']
        elif type == 2:
            content = data['content2']
            content_mask = data['content_mask2']
            path_map = data['path_map2']
            paths = data['paths2']
            paths_mask = data['paths_mask2']
            r_paths = data['r_paths2']
            r_paths_mask = data['r_paths_mask2']
            r_path_idx = data['r_path_idx2']
            named = data['named2']


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
        memory, memory_key_padding_mask = self.encode(data,1)
        memory2, memory_key_padding_mask2 = self.encode(data,2)
        code_vec_atten = torch.mean(memory, dim=1)
        code_vec_atten2 = torch.mean(memory2, dim=1)

        # code_vec_atten = memory[:,0,:]
        # code_vec_atten2 = memory2[:,0,:]

        abs_dist = torch.abs(torch.add(self.fc(code_vec_atten),-self.fc(code_vec_atten2)))
        out = torch.sigmoid(self.hidden2label(abs_dist)).squeeze(1)
        return out


