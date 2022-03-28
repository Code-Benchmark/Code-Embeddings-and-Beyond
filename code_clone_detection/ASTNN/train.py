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

from gensim.models.word2vec import Word2Vec
word2vec = Word2Vec.load("./train/embedding/node_w2v_128_new").wv
word2vec.index2word

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
    print("Train for ", str.upper(lang))
    train_data = pd.read_pickle(root+lang+'/train/blocks_new.pkl').sample(frac=1)
    train_data = train_data.replace(-1, 0)
    val_data = pd.read_pickle(root+lang+'/dev/blocks_new.pkl').sample(frac=1)
    val_data = val_data.replace(-1, 0)
    test_data = pd.read_pickle(root+lang+'/test/blocks_new.pkl').sample(frac=1)
    test_data = test_data.replace(-1, 0)

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

    print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    for t in range(5, categories+1):
        train_data_t, val_data_t, test_data_t = train_data, val_data, test_data
        # training procedure

        train_loss_ = []
        val_loss_ = []

        for epoch in range(EPOCHS):
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            predicts = []
            trues = []
            bs = BATCH_SIZE
            # while i < len(train_data_t):
            for i in tqdm(range(0, len(train_data_t), bs)):
                if i + bs > len(train_data_t):
                    bs = len(train_data_t) - i
                batch = get_batch(train_data_t, i, bs)
                # i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()

                optimizer.step()

                total += len(train_labels)
                total_loss += loss.item() * len(train_labels)
                predicted = (output.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                trues.extend(train_labels.cpu().numpy())

            train_loss_.append(total_loss / total)
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

            total_loss = 0.0
            total = 0.0
            i = 0
            bs = BATCH_SIZE
            predicts = []
            trues = []
            # while i < len(val_data_t):
            #     batch = get_batch(val_data_t, i, BATCH_SIZE)
            #     i += BATCH_SIZE
            for i in tqdm(range(0, len(val_data_t), bs)):
                if i + bs > len(val_data_t):
                    bs = len(val_data_t) - i
                batch = get_batch(val_data_t, i, BATCH_SIZE)
                val1_inputs, val2_inputs, val_labels = batch
                if USE_GPU:
                    val1_inputs, val2_inputs, val_labels = val1_inputs, val2_inputs, val_labels.cuda()

                model.batch_size = len(val_labels)
                model.hidden = model.init_hidden()
                output = model(val1_inputs, val2_inputs)

                loss = loss_function(output, Variable(val_labels))

                total += len(val_labels)
                total_loss += loss.item() * len(val_labels)
                predicted = (output.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                trues.extend(val_labels.cpu().numpy())

            val_loss_.append(total_loss / total)
            precision_, recall_, f1_, _ = precision_recall_fscore_support(trues, predicts, average='binary')

            print('categories-%d  [Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  % (t, epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch]))
            print("Train results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
            print("Dev results(P,R,F1):%.3f, %.3f, %.3f" % (precision_, recall_, f1_))

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()
                        }, PATH)


