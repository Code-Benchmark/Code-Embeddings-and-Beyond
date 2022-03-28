import pickle
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
from tqdm import tqdm
def num_docstring(datapath):
    print('embedding docstring...')
    word2vec = Word2Vec.load("data/node_wiz_token_w2v_128").wv
    in2en = word2vec.index_to_key
    data = pd.read_csv(datapath)
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
                doc_num.append(-1)
        num.append(doc_num)

        data['docstring'][i] = doc_num
    return data


def read_treecaps_data(codepath):
    # data = pickle.load(open(datapath, 'rb'))
    code = num_docstring(codepath)
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

    return train_d, test_d