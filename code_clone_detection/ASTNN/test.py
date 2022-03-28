import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
warnings.filterwarnings('ignore')



def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_ids_x'])
        x2.append(item['code_ids_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument('--lang')
    args = parser.parse_args()
    args.lang = 'java'
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = 'data/'

    lang = args.lang
    categories = 1
    if lang == 'java':
        categories = 5

    test_data = pd.read_pickle(root+lang+'/test/blocks_new.pkl').sample(frac=1)

    word2vec = Word2Vec.load(root+lang+"/train/embedding/node_w2v_128_new").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 64
    USE_GPU = True

    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()
    PATH = './model/model_clone_java_new.pkl'
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])


    test_data_t = test_data

    print("Testing..." )
    # testing procedure
    predicts = []
    trues = []
    total_loss = 0.0
    total = 0.0
    i = 0
    for i in tqdm(range(0, len(test_data_t), BATCH_SIZE)):
        if i + BATCH_SIZE > len(test_data_t):
            BATCH_SIZE = len(test_data_t) - i
        batch = get_batch(test_data_t, i, BATCH_SIZE)
        i += BATCH_SIZE
        test1_inputs, test2_inputs, test_labels = batch
        if USE_GPU:
            test_labels = test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test1_inputs, test2_inputs)

        predicted = (output.data > 0.5).cpu().numpy()
        predicts.extend(predicted)
        trues.extend(test_labels.cpu().numpy())


    result = pd.DataFrame(np.array(predicts), columns=['predict'])
    result['true'] = pd.DataFrame(np.array(trues))
    result['label'] = pd.DataFrame(np.array(trues))
    result.loc[result['label'] > 0, 'label'] = 1
    weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
    precision, recall, f1 = 0, 0, 0
    for k in range(1, categories+1):
        trues_ = result[result['true'].isin([0, k])]['label'].values
        predicts_ = result[result['true'].isin([0, k])]['predict'].values
        p, r, f, _ = precision_recall_fscore_support(trues_, predicts_, average='binary')
        precision += weights[k] * p
        recall += weights[k] * r
        f1 += weights[k] * f
        print("Type-" + str(k) + ": " + str(p) + " " + str(r) + " " + str(f))

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))