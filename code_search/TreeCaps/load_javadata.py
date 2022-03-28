import pickle
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
from tqdm import tqdm
def num_docstring(root, datapath):
    print('embedding docstring...')
    word2vec = Word2Vec.load(root+"/node_wiz_token_w2v_128").wv
    in2en = word2vec.index_to_key
    data = pd.read_csv(datapath)
    #print(data{1})
    num = []
    k = 0

    for i in tqdm(range(len(data))):

        docstring = (data['docstring'][i].replace("\n", "")).split(' ')
        doc_num = []
        for j in range(len(docstring)):
            if docstring[j] in in2en:
                doc_num.append(in2en.index(docstring[j]))
            else:
                k = k + 1
                # print(docstring[j])
                doc_num.append(-1)
        num.append(doc_num)

        data['docstring'][i] = doc_num
    # path = 'nlp_index.pkl'
    # with open(path, 'wb') as fh:
    #     pickle.dump(data,fh)
    # data = pd.read_pickle(path)
    return data


def read_train_data(root,datapath, codepath):
    # data = pickle.load(open(datapath, 'rb'))
    code = num_docstring(root,codepath)
    code.columns = ['id1','id','docstring', 'code', 'partition']
    codedata = code.drop('code', axis=1)
    codedoc = code['docstring']
    doc = codedoc.sample(frac=1)
    codedata['docstring2'] = list(doc)

    for i in range(0, len(code) - 1):
        if code['partition'][i] != 'train':
            print("i:")
            print(i)
            break

    train_data = codedata.head(i)
    test_data = codedata.tail(len(code) - i)
    print('len:',len(test_data))
    train_d = train_data.set_index('id').T.to_dict('list')
    test_d = test_data.set_index('id').T.to_dict('list')
    all_d = codedata.set_index('id').T.to_dict('list')
    return train_d, test_d, all_d

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
def get_testbatch(allnodes, allchildren, idx, batch_size, all_csv,data,trainlen):
    # allnodes, allchildren, allabels = gen
    node1,ch1,doc1,doc2= [],[],[],[]
    tmp = data[idx:idx+batch_size]
    i = 0
    for _, t in tmp:

        nodes, children = getid(idx+i+trainlen, allnodes, allchildren)
        node1.append(nodes)
        ch1.append(children)
        docs1 = all_csv[data[idx + i]['id']][1]
        docs2 = all_csv[data[idx + i]['id']][3]
    
        doc1.append(docs1)
        doc2.append(docs2)
        i = i + 1


    node1, ch1 = _pad_batch(node1, ch1)

    return node1,node1,ch1,ch1,doc1,doc2

def _pad_batch(nodes, children):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0])
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



