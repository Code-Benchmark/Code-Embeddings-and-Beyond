from model import Code2SeqCloneDetector
from process_data import C_code
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import json
import random
import csv
import os

data_json = json.loads(open('./data/java/code_data.json', 'r').read())

def get_data(x,  max_contexts):
    x_code_true = []
    for id in x:
        x_code_true.append(data_json[str(id)])
    x_code_true = np.array(x_code_true).swapaxes(1, 2)
    starts = torch.LongTensor(x_code_true[:, 0, :].tolist()).cuda()
    paths = torch.LongTensor(x_code_true[:, 1, :].tolist()).cuda()
    ends = torch.LongTensor(x_code_true[:, 2, :].tolist()).cuda()

    return starts, paths, ends

if __name__ == '__main__':
    BATCH_SIZE = 64
    t = 5



    train_data = pd.read_csv('./data/java/train_data.csv')
    train_data = train_data.replace(-1, 0)
    # train_data = train_data[0:100]
    dev_data = pd.read_csv('./data/java/dev_data.csv')
    dev_data = dev_data.replace(-1, 0)
    test_data = pd.read_csv('./data/java/test_data.csv')
    test_data = test_data.replace(-1, 0)
    test_data.loc[test_data['label'] > 0, 'label'] = 1

    train_small_num = 100
    train_num = 901724
    dev_num = 416328
    test_num = 416328


    # Load Model
    nodes_dim = 20659  # c:2021
    paths_dim = 178     # c:166
    embedding_dim = 128
    code_vector_size = 4 * embedding_dim
    dropout = 0.25
    USE_GPU = True

    model = Code2SeqCloneDetector(nodes_dim+1, paths_dim+1, embedding_dim, code_vector_size, dropout)

    if USE_GPU:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.BCELoss()
    print('Done creating Code2SeqModel')

    EPOCHS = 200
    # writer = SummaryWriter('log/0112')
    PATH = './model/java/clone_model.pkl'
    # checkpoint = torch.load(PATH)
    # start_epoch = checkpoint['epoch']
    # model.load_state(checkpoint['model_state_dnict'])

    print('Start training...')
    max_contexts = 500
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    for epoch in range(EPOCHS):
        # train
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        predicts = []
        trues = []
        bs = BATCH_SIZE
        model.train()
        for i in tqdm(range(0, train_num, BATCH_SIZE)):
            if i+bs > train_num:
                bs = train_num - i

            # context_x_batch, context_y_batch = data.load_data('train', i, bs)
            # y_batch = data.get_label('train', i, bs)
            batch_data = train_data[i:i+bs]
            train_labels = torch.FloatTensor(batch_data['label'].values.tolist()).cuda()

            train_starts_x1, train_paths_x1, train_ends_x1 = get_data(batch_data['id1'].values.tolist(), max_contexts)
            train_starts_x2, train_paths_x2, train_ends_x2 = get_data(batch_data['id2'].values.tolist(), max_contexts)
            # model.train()
            optimizer.zero_grad()

            output = model(train_starts_x1, train_paths_x1, train_ends_x1,
                           train_starts_x2, train_paths_x2, train_ends_x2)

            loss = loss_function(output, train_labels)

            loss.backward()
            optimizer.step()

            # calc training
            # _, predicted = torch.max(output.data, 1)
            # total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_labels)

            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(train_labels.cpu().numpy())
            #print("loss: %f   acc: %f" % (loss.item(), (predicted == train_labels).sum().float()/len(train_labels)))

        train_loss_.append(total_loss / total)
        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

        # validation
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        predicts = []
        trues = []
        bs = BATCH_SIZE
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, dev_num, BATCH_SIZE)):
                if i + bs > dev_num:
                    bs = dev_num - i

                batch_data = dev_data[i:i+bs]
                val_labels = torch.FloatTensor(batch_data['label'].values.tolist()).cuda()

                val_starts_x1, val_paths_x1, val_ends_x1 = get_data(batch_data['id1'].values.tolist(), max_contexts)
                val_starts_x2, val_paths_x2, val_ends_x2 = get_data(batch_data['id2'].values.tolist(), max_contexts)

                output = model(val_starts_x1, val_paths_x1, val_ends_x1,
                               val_starts_x2, val_paths_x2, val_ends_x2 )
                loss = loss_function(output, val_labels)

                total += len(val_labels)
                total_loss += loss.item() * len(val_labels)
                predicted = (output.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                trues.extend(val_labels.cpu().numpy())

            val_loss_.append(total_loss / total)
            precision_, recall_, f1_, _ = precision_recall_fscore_support(trues, predicts, average='binary')


            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f'
                  % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch]))

            print("Train results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
            print("Dev results(P,R,F1):%.3f, %.3f, %.3f" % (precision_, recall_, f1_))
            #
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()
                        }, PATH)
            #
