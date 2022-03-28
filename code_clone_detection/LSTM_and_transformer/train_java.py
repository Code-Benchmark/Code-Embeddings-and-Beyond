import pandas as pd
import random
import torch
from tqdm import tqdm
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import CloneDetector
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import os
import sys

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(eval(item['code_ids_x']))
        x2.append(eval(item['code_ids_y']))
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)

if __name__ == '__main__':
    root = 'data/'
    lang = 'java'
    train_data = pd.read_csv(root+'sample_train_new.csv')
    train_data = train_data.replace(-1, 0)
    train_data = train_data
    val_data = pd.read_csv(root + 'sample_dev_new.csv')
    val_data = val_data.replace(-1, 0)
    test_data = pd.read_csv(root+'sample_test_new.csv')
    test_data = test_data.replace(-1, 0)

    EPOCHS = 10
    BATCH_SIZE = 64
    USE_GPU = True
    EMBEDDING_DIM = 200
    MAX_Length = 1000
    MAX_TOKENS = 17579  # Java：85104   c:10842
    categories = 5
    model = 'transformer'

    model = CloneDetector(MAX_TOKENS+1, EMBEDDING_DIM, MAX_Length, model)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()
    PATH = './model/model_clone_transformer_java_new.pkl'

    train_loss_ = []
    val_loss_ = []

    print('Start training...')
    # training procedure
    for t in range(5, categories+1):
        train_data_t, val_data_t, test_data_t = train_data, val_data, test_data

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
            model.train()
            # while i < len(train_data_t):
            for i in tqdm(range(0, len(train_data_t), bs)):
                if i + bs > len(train_data_t):
                    bs = len(train_data_t) - i
                batch = get_batch(train_data_t, i, bs)
                # i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                optimizer.zero_grad()
                output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, train_labels)
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
            model.eval()
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

                output = model(val1_inputs, val2_inputs)

                loss = loss_function(output, val_labels)

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
