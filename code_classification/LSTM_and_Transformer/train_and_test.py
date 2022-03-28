import pandas as pd
import random
import torch
from tqdm import tqdm
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import transformer_model
from model import lstm_model
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(eval(item['code_ids']))
        labels.append(item['label']-1)
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    root = 'data/'
    train_data = pd.read_csv(root+'sample_train.csv')
    val_data = pd.read_csv(root + 'sample_dev.csv')
    test_data = pd.read_csv(root+'sample_test.csv')

    LABELS = 104
    EPOCHS = 20
    BATCH_SIZE = 64
    USE_GPU = True
    EMBEDDING_DIM = 200
    MAX_TOKENS = 12097 

    #model = lstm_model(MAX_TOKENS + 1, EMBEDDING_DIM)
    model = transformer_model(MAX_TOKENS+1, EMBEDDING_DIM)
    # PATH = './model/model_lstm.pkl'
    PATH = './model/model_transformer.pkl'
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')

    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()
        # training
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        bs = BATCH_SIZE
        for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
            if i+bs > len(train_data):
                bs = len(train_data) - i
            batch = get_batch(train_data, i, bs)
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            output = model(train_inputs)

            loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)


        # validation
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        bs = BATCH_SIZE
        for i in tqdm(range(0, len(val_data), BATCH_SIZE)):
            if i+bs > len(val_data):
                bs = len(val_data) - i
            batch = get_batch(val_data, i, bs)
            # i += BATCH_SIZE
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            output = model(val_inputs)

            loss = loss_function(output, val_labels)

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
