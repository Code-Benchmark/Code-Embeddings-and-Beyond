import pandas as pd
import torch
from tqdm import tqdm
import time
from model import GGNN

import json

data_json = json.loads(open('data/sample_ggnn.json', 'r').read())

if __name__ == '__main__':
    data = pd.read_pickle('../data.pkl')
    train_data = data[data['partition'] =='train']
    val_data = data[data['partition'] == 'dev']
    test_data = data[data['partition'] == 'test']

    LABELS = 104
    EPOCHS = 20
    BATCH_SIZE = 64
    USE_GPU = True
    vocablen = 15388
    edge_type = 7
    EMBEDDING_DIM = 200
    num_layers = 4
    device = torch.device('cuda:0')

    model = GGNN(vocablen, EMBEDDING_DIM, num_layers, device)
    PATH = './model/model.pkl'
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
        # train
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        bs = BATCH_SIZE
        model.train()
        for i in tqdm(range(0, len(train_data))):
            # batchsize = 1
            train_label = train_data[i:i+1]['label'].values[0]
            train_label = torch.LongTensor([train_label-1]).cuda()
            train_inputs = data_json[str(train_data[i:i+1]['id'].values[0])]

            model.zero_grad()
            model.batch_size = 1
            output = model(train_inputs)

            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_label)
            total += 1
            total_loss += loss.item()

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)


        # eval
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        bs = BATCH_SIZE
        model.eval()
        for i in tqdm(range(0, len(val_data))):
            val_label = val_data[i:i + 1]['label'].values[0]
            val_label = torch.LongTensor([val_label - 1]).cuda()
            val_inputs = data_json[str(val_data[i:i + 1]['id'].values[0])]

            model.batch_size = 1
            output = model(val_inputs)

            loss = loss_function(output, val_label)

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_label)
            total += 1
            total_loss += loss.item()
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

    # test
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    for i in tqdm(range(0, len(test_data))):
        test_label = test_data[i:i + 1]['label'].values[0]
        test_label = torch.LongTensor([test_label - 1]).cuda()
        test_inputs = data_json[str(test_data[i:i + 1]['id'].values[0])]
        model.batch_size = 1
        output = model(test_inputs)

        loss = loss_function(output, test_label)

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_label)
        total += 1
        total_loss += loss.item()
    print("Testing results(Acc):", total_acc.item() / total)
