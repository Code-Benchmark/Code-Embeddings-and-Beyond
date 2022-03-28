import torch
from model_for_attr import SearchModel
from gensim.models.word2vec import Word2Vec
import pandas as pd
# from gensim.models.word2vec import Word2Vec
import numpy as np
import argparse
import json
from captum.attr import IntegratedGradients, LayerIntegratedGradients

if __name__ == '__main__':
    data_json = json.loads(open('./data/ggnn_wiz_w2v.json', 'r').read())

    word2vec_doc = Word2Vec.load("./data/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec_doc.syn0.shape[0]
    EMBEDDING_DIM = word2vec_doc.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec_doc.syn0.shape[0]] = word2vec_doc.syn0

    EPOCHS = 100
    BATCH_SIZE = 64
    USE_GPU = True
    vocablen = MAX_TOKENS + 1  # 1173693 #1145175
    edge_type = 7
    EMBEDDING_DIM = 128
    num_layers = 4
    device = torch.device('cuda:0')
    # MAX_TOKENS = 12097
    # def __init__(self, vocablen, embedding_dim, num_layers, device):
    model = SearchModel(vocablen, EMBEDDING_DIM, num_layers, device, embeddings)
    if USE_GPU:
        model.cuda()


    PATH = './model/model_search_0316.pkl'
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

    data = pd.read_csv('./data/new_data_0316.csv')
    data = data[data['partition']=='test']
    data = data.reset_index()
    res = []
    # set = np.load("../../%d_999.npy" % id)
    # for case in set:
    i= 0
    for index, row in data.iterrows():
        if i %1000 == 0:
            print(i)
    #     test = data_test[data_test['index'] == case]
        data_x = data_json[str(row['index'])]
        node_text_x = torch.LongTensor(data_x['node_ids']).cuda()
        node_text_x = model.encoder.embed(node_text_x)
        edge_index_x = torch.LongTensor(data_x['edges']).cuda()
        edge_attr_x = torch.LongTensor(data_x['edge_types']).cuda()
        edge_attr_x = model.encoder.edge_embed(edge_attr_x - 1).mean(1)

        x1 = model.encoder.ggnnlayer(node_text_x, edge_index_x, edge_attr_x)

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
        code_vec, nl_vec = model(x1, nl_emb)

        i = i + 1
        res.append([row['index'], code_vec.detach().cpu().numpy()[0], nl_vec.detach().cpu().numpy()[0]])
    # break
    res = pd.DataFrame(res, columns=["id", "code_vec", "nl_vec"])
    res.to_pickle("ggnn_vectors.pkl")

