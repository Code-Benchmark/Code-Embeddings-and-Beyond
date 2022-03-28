import pickle
import pandas as pd
def load_tbcnn_data(rootfile, treefile, embeddingfile,type):
    trees_file = rootfile + treefile
    if type == 'train':
        with open(trees_file, 'rb') as fh:
            trees, _,_, label = pickle.load(fh)
    elif type == 'dev':
        with open(trees_file, 'rb') as fh:
            _, trees,_, label = pickle.load(fh)
    elif type == 'test':
        with open(trees_file, 'rb') as fh:
            _, _,trees, label = pickle.load(fh)
    label = list(set(label))
    embedding_file = rootfile + embeddingfile
    with open(embedding_file, 'rb') as fh:
        embeddings,lookup = pickle.load(fh)
        num_feats = len(embeddings[0])
    return trees, label, embeddings, lookup, num_feats

def load_treecaps_data(rootfile, treefile, lookupfile,type):
    trees_file = rootfile + treefile
    if type == 'train':
        with open(trees_file, 'rb') as fh:
            trees, _, _, label = pickle.load(fh)
    elif type == 'dev':
        with open(trees_file, 'rb') as fh:
            _, trees, _, label = pickle.load(fh)
    elif type == 'test':
        with open(trees_file, 'rb') as fh:
            _, _, trees, label = pickle.load(fh)
    label = list(set(label))

    embedding_file = rootfile + lookupfile
    NODE_LIST = pd.read_pickle(embedding_file)
    NODE_LIST.append('UNK')
    NODE_MAP = {x: i for (i, x) in enumerate(NODE_LIST)}

    return trees, label, NODE_MAP