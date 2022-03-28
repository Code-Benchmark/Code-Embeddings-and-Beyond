import pandas as pd
import torch
import time
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from captum.attr import IntegratedGradients
from model.model_for_Attr import SearchModel
from model.loss import RankLoss

from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    root = 'data/'
    lang = 'java'

    word2vec = Word2Vec.load("./data/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    USE_GPU = True
    code_model = "Transformer"
    # writer = SummaryWriter('log/0125')

    model = SearchModel(code_model, EMBEDDING_DIM, MAX_TOKENS+1, USE_GPU)
    model.cuda()
    model.eval()
    # decay_ratio = 0.95

    PATH = './models/model_Transformer_0318.pkl'
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])

  
    test_data = pd.read_csv('./data/test_data_0318.csv')
    test_data = test_data.reset_index()
   #  test = test_data[test_data['index']==13097]
    code_vecs = []
    nl_vecs = []
    i = 1
    res = []
    for index, row in test_data.iterrows():
        # test = test_data[test_data['index'] == case]
        if i % 1000 == 0:
            print(i)
        i = i + 1
        input_x_code = eval(row['code_ids_'])
        seq = [input_x_code]
        encodes = torch.LongTensor(seq).cuda()
        code_emb = model.embedding_layer(encodes + 2)

        input_x_doc = eval(row['doc_ids_'])
        max_len_d = 20
        seq_d = []
        if len(input_x_doc) > max_len_d:
            seq_d.append(input_x_doc[0: max_len_d])
        else:
            seq_d.append(
                np.pad(np.array(input_x_doc), (0, max_len_d - len(input_x_doc)), 'constant',
                       constant_values=-1).tolist())
        encodes_d = torch.LongTensor(seq_d).cuda()
        nl_emb = model.embedding_layer(encodes_d + 2)
        # print(model(code_emb, nl_emb))
        code_vec, nl_vec = model(code_emb, nl_emb)
        # code_vecs.extend(code_vec.cpu().numpy())
        # nl_vecs.extend(nl_vec)

        res.append([row['index'], code_vec.detach().cpu().numpy()[0], nl_vec.detach().cpu().numpy()[0]])
        # break
    res = pd.DataFrame(res, columns=["id", "code_vec","nl_vec"])
    res.to_pickle("transformer_vectors.pkl")

