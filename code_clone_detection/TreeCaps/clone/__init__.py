import pickle
import pandas as pd
from gensim.models.word2vec import Word2Vec
import torch
from gensim.models import word2vec
from tqdm import tqdm
def num_docstring(csv,ids,word2vec):
    print('embedding docstring...')
    # in2en = word2vec.index_to_key
    # datas = csv
    # #print(data{1})
    # num = []
    # k = 0
    # data = datas.iloc[ids].values
    # data = pd.DataFrame(data)
    # data = data.T
    # data.columns = ['id', 'docstring', 'code', 'partition']
    #
    # data['docstring2'] = datas['docstring'][ids+1]
    # docstring2 = (datas['docstring'][ids+1].replace("\n", "")).split(' ')
    # docstring = (datas['docstring'][ids].replace("\n", "")).split(' ')
    # doc_num = []
    # for j in range(len(docstring)):
    #     if docstring[j] in in2en:
    #         doc_num.append(in2en.index(docstring[j]))
    #     else:
    #         k = k + 1
    #         # print(docstring[j])
    #         doc_num.append(-1)
    # # num.append(doc_num)
    # data['docstring'][0] = doc_num
    #
    # doc_num = []
    # for j in range(len(docstring2)):
    #     if docstring2[j] in in2en:
    #         doc_num.append(in2en.index(docstring2[j]))
    #     else:
    #         k = k + 1
    #         # print(docstring[j])
    #         doc_num.append(-1)
    # # num.append(doc_num)
    # data['docstring2'][0] = doc_num


    data = torch.load('../tbcnn/attr_analysis/search/result/torchdata.npy')
    return data


def read_train_data(datapath, csv, ids,word2vec):
    # data = pickle.load(open(datapath, 'rb'))
    code = num_docstring(csv, ids,word2vec)
    code.columns = ['id', 'docstring', 'code', 'partition', 'docstring2']
    codedata = code.drop('code', axis=1)
    # codedoc = code['docstring']
    # doc = codedoc.sample(frac=1)
    # codedata['docstring2'] = code['docstring']

    # for i in range(0, len(code) - 1):
    #     if code['partition'][i] != 'train':
    #         print("i:")
    #         print(i)
    #         break
    #
    # for j in range(1, len(data) - 1):
    #     if data[j]['id'] == 0:
    #         print("j:")
    #         print(j)
    #         break
    #
    # train_data = codedata.head(i)
    # test_data = codedata.tail(len(code) - i)
    # print('len:',len(test_data))
    # train = data[:j]
    # test = data[j + 1:]
    # train_d = train_data.set_index('id').T.to_dict('list')
    # test_d = test_data.set_index('id').T.to_dict('list')
    # all_d = codedata.set_index('id').T.to_dict('list')
    return codedata

def load_data(rootfile, treefile, embeddingfile):
    trees_file = rootfile + treefile

    with open(trees_file, 'rb') as fh:
        trees = pickle.load(fh)

    # id = list(set(id))
    embedding_file = rootfile + embeddingfile
    with open(embedding_file, 'rb') as fh:
        embeddings,lookup = pickle.load(fh)
        num_feats = len(embeddings[0])
    return trees, embeddings, lookup, num_feats

def get_batch(allnodes, allchildren, idx, batch_size, train_csv,data):
    # allnodes, allchildren, allabels = gen
    node1,ch1,doc1,doc2= [],[],[],[]
    tmp = data[idx:idx+batch_size]
    i = 0
    for _, t in tmp:

        nodes, children = getid(idx+i, allnodes, allchildren)
        node1.append(nodes)
        ch1.append(children)
        docs1 = train_csv[data[idx+i]['id']][1]
        docs2 = train_csv[data[idx + i]['id']][3]
        doc1.append(docs1)
        doc2.append(docs2)
        i = i + 1


    node1, ch1 = _pad_batch(node1, ch1)

    return node1,node1,ch1,ch1,doc1,doc2
def get_testbatch(allnodes, allchildren, idx, codedata):
    # allnodes, allchildren, allabels = gen
    node1,ch1,doc1,doc2= [],[],[],[]
    # tmp = data[idx:idx+batch_size]

    nodes, children = getid(0, allnodes, allchildren)
    node1.append(nodes)
    ch1.append(children)
    docs1 = codedata['docstring'][0]
    docs2 = codedata['docstring2'][0]
    doc1.append(docs1)
    doc2.append(docs2)

    node1, ch1 = _pad_batch(node1, ch1)

    return node1,node1,ch1,ch1,doc1,doc2
def _pad_batch(nodes, children):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children

def getid(ids,allnodes, allchildren):
    nodes, children = [], []
    n, c = allnodes, allchildren

    nodes = n[ids]
    children = c[ids]

    return nodes, children

def merge(pair,data):
    data['index'] = range(len(data['id']))

    pair['id1'] = pair['id1'].astype(int)
    pair['id2'] = pair['id2'].astype(int)
    df = pd.merge(pair, data, how='left', left_on='id1', right_on='id')
    df = pd.merge(df, data, how='left', left_on='id2', right_on='id')

    df.drop(['id_x', 'id_y', 'code_x','code_y'], axis=1, inplace=True)
    df.dropna(inplace=True)

    return df

