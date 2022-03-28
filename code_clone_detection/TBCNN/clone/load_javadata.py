import pickle
import pandas as pd
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

def get_batch(allnodes, allchildren,idx,batch_size,data):
    # allnodes, allchildren, allabels = gen
    node1,node2,ch1,ch2, label = [],[],[],[],[]
    tmp = data.iloc[idx:idx+batch_size]
    for _, t in tmp.iterrows():
        # nodes, children = getid(int(t['index_x']), allnodes, allchildren)
        nodes, children = allnodes[int(t['index_x'])], allchildren[int(t['index_x'])]
        node1.append(nodes)
        ch1.append(children)

        # nodes, children = getid(int(t['index_y']), allnodes, allchildren)
        nodes, children = allnodes[int(t['index_y'])], allchildren[int(t['index_y'])]
        node2.append(nodes)
        ch2.append(children)
        if t['label'] == -1:
            label.append(int(0))

        else:
            label.append(int(t['label']))
    node1, ch1 = _pad_batch(node1, ch1)
    node2, ch2 = _pad_batch(node2, ch2)
    return node1,node2,ch1,ch2, label

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


# datapath = r'E:\科研\复现\TBCNN\3\tbcnn-data\clonedata\java_new\codes.csv'
# pairpath = r'E:\科研\复现\TBCNN\3\tbcnn-data\clonedata\java_new\dev_idpairs.csv'
#
# data = pd.read_csv(datapath)
# pair = pd.read_csv(pairpath)
