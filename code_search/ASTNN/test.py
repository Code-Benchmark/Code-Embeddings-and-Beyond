import pandas as pd
import torch
import time
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from model_for_Attr import ASTNN4Search
from model import RankLoss
import random
import tensorflow as tf
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from captum.attr import IntegratedGradients
warnings.filterwarnings('ignore')

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2 = [], []
    fp_x1, fp_x2 = [], []

    for _, item in tmp.iterrows():
        x1.append(item['code_ids'])
        x2.append(eval(item['doc_ids']))
        fp_x1.append(item['code_ids'])
        fp_x2.append(eval(item['fp_doc_ids']))
    return x1, x2, fp_x1, fp_x2    # , torch.FloatTensor(labels)



if __name__ == '__main__':
    root = 'data/'
    lang = 'java'

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    EPOCHS = 100
    BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    # MAX_TOKENS_doc = 18912
    USE_GPU = True

    # f = open("result.txt", "a")
    EMBEDDING_DIM = 128
    EMBEDDING_DIM_doc = 128
    # MAX_TOKENS_code = 115061  #49429 #18898    #49429
    # MAX_TOKENS_doc = 115061  #49429 #18898  #49429

    word2vec = Word2Vec.load(root+lang+"/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    model = ASTNN4Search(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS+1,  # +1 because nan = max_token
                            ENCODE_DIM, BATCH_SIZE, USE_GPU, embeddings)
    model.cuda()

    PATH = './model/model_0304.pkl'
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])

    res = []
    test_data = pd.read_pickle(root+lang+'/test_data_0304.pkl')
    test_data = test_data.reset_index()
    # test = test_data[test_data['index'] == 339161]# 13918  15966  9188
    i = 0
    for index, row in test_data.iterrows():
        if i % 1000 == 0:
            print(i)
        i = i + 1
        input_code_ = row["code_ids"]
        encodes_x = []
        lens_x = len(input_code_)
        for j in range(lens_x):
            encodes_x.append(input_code_[j])
        input_code = model.encoder(encodes_x, lens_x)
        input_code = input_code.unsqueeze(0)

        input_x_doc = eval(row['doc_ids'])
        max_len_d = 20
        seq_d = []
        if len(input_x_doc) > max_len_d:
            seq_d.append(input_x_doc[0: max_len_d])
        else:
            seq_d.append(
                np.pad(np.array(input_x_doc), (0, max_len_d - len(input_x_doc)), 'constant', constant_values=-1).tolist())
        encodes_d = torch.LongTensor(seq_d).cuda()
        nl_emb = model.encoder_doc.embedding(encodes_d + 1)

        code_vec, nl_vec = model(input_code, nl_emb)
        # print(model(code_embedded,nl_emb))
        res.append([row['index'], code_vec.detach().cpu().numpy()[0], nl_vec.detach().cpu().numpy()[0]])
    res = pd.DataFrame(res, columns=["id", "code_vec", "nl_vec"])
    res.to_pickle("astnn_vectors.pkl")

