import torch
from model_for_attr import CodeSearchModel
import pandas as pd
from gensim.models.word2vec import Word2Vec
# from gensim.models.word2vec import Word2Vec
import numpy as np
import argparse
from captum.attr import IntegratedGradients

# path_dict = np.load('./data/path_dict.npy', allow_pickle=True).item()
token_dict = np.load('./data/new_token_id_dict.npy', allow_pickle=True).item()
def load_data(row):
    source_tokens = []
    path_tokens = []
    target_tokens = []
    context_valid_masks = []
    cur_source_tokens = []
    cur_path_tokens = []
    cur_target_tokens = []
    cur_context_masks = []
    line = row.split(' ')

    cnt = 0
    for path in line[1:]:
        source_token, path_token, target_token = map(int, path.strip().split(','))
        source_token = token_dict[source_token]
        target_token = token_dict[target_token]
        cur_source_tokens.append(source_token)
        cur_path_tokens.append(path_token)
        cur_target_tokens.append(target_token)
        cur_context_masks.append(1)
        cnt = cnt + 1
   
    source_tokens.append(cur_source_tokens)
    path_tokens.append(cur_path_tokens)
    target_tokens.append(cur_target_tokens)
    context_valid_masks.append(cur_context_masks)

    return source_tokens, path_tokens, target_tokens, context_valid_masks

if __name__ == '__main__':
    word2vec = Word2Vec.load("./data/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    nodes_dim = 999296  # 999178
    paths_dim = 304280
    embedding_dim = EMBEDDING_DIM
    code_vector_size = 3 * embedding_dim
    # output_dim = 2
    dropout = 0.25
    USE_GPU = True
    max_token = MAX_TOKENS
    # max_path_node = 208
    max_nl_len = 20
    # nodes_dim, paths_dim, embedding_dim, code_vector_size, vocab_size_doc, vocab_size_path=None, pretrained_weight
    model = CodeSearchModel(nodes_dim, paths_dim, embedding_dim, code_vector_size, max_token + 1, embeddings)
    if USE_GPU:
        model.cuda()
    model.eval()
    PATH = './model/model_0304.pkl'
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    # dev = pd.read_csv('./data/new_test_data.csv', header=None)
    # dev.columns = ["id", "context", "doc_ids", "fp_doc_ids"]
    # x_code_val = data.load_data('test')
    data_test = pd.read_csv('./data/test_data_0304.csv')
    data_test.columns = ["id","code_ids","doc_ids","fp_doc_ids"]
    def s(k):
        return k.split(" ")[0].split("/")[-1][:-5]
    data_test["id"] = data_test["code_ids"].apply(s)
    data_test = data_test.reset_index()
    data_test.columns = ["index", "id", "code_ids", "doc_ids", "fp_doc_ids"]
    # data_test = data_test[data_test["label"]==5]
    id = 343064
    # set.append(id)
    res = []
    # set = np.load("../../%d_999.npy" % id)
    # for case in set:
    i = 0
    for index, row in data_test.iterrows():
        if i % 1000 == 0:
            print(i)
        i = i + 1
    #     test = data_test[data_test['index'] == case]
        input_x_o = row["code_ids"]
        # input_x_o = test['context_x'].values[0]
        starts_x1_o, paths_x1_o, ends_x1_o, masks_x1_o = load_data(input_x_o)
        starts_embedded = model.encoder.node_embedding(torch.LongTensor(starts_x1_o).cuda())
        ends_embedded = model.encoder.node_embedding(torch.LongTensor(ends_x1_o).cuda())
        paths_embedded = model.encoder.path_embedding(torch.LongTensor(paths_x1_o).cuda())
        code_embedded = torch.cat((starts_embedded, paths_embedded, ends_embedded), dim=2)

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
        code_vec, nl_vec = model(code_embedded, nl_emb)
        # print(model(code_embedded,nl_emb))
        res.append([row['index'], code_vec.detach().cpu().numpy()[0], nl_vec.detach().cpu().numpy()[0]])
    res = pd.DataFrame(res, columns=["id", "code_vec", "nl_vec"])
    res.to_pickle("code2vec_vectors.pkl")
