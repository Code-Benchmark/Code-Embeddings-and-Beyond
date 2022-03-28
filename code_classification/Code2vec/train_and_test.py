from model import Code2Vec
from process_data import C_code
import torch
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os


if __name__ == '__main__':
    BATCH_SIZE = 64
    # Load Data
    data = C_code()

    x = data.load_data('train')
    y = data.get_label('train')
    print('train dataset: ' + str(np.array(x).shape))

    x_val = data.load_data('valid')
    y_val = data.get_label('valid')
    print('valid dataset: ' + str(np.array(x_val).shape))

    x_test = data.load_data('test')
    y_test = data.get_label('test')
    print('test dataset: ' + str(np.array(x_test).shape))

    print('Done Data Loader')

    # Load Model
    nodes_dim = 9999
    paths_dim = 183761
    embedding_dim = 128
    code_vector_size = 3 * embedding_dim
    output_dim = 104
    dropout = 0.50  # 0.25
    USE_GPU = True

    model = Code2Vec(nodes_dim, paths_dim, embedding_dim, code_vector_size, output_dim, dropout)

    if USE_GPU:
        model.cuda()
    decay_ratio = 0.95
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()
    print('Done creating Code2VecModel model')

    EPOCHS = 100
    writer = SummaryWriter('log')
    PATH = './model/c2v_clf_model'
    # checkpoint = torch.load(PATH)
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['model_state_dict'])

    print('Start training...')
    bs = BATCH_SIZE
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()
        # train
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for i in tqdm(range(0, len(y), BATCH_SIZE)):
            if i+bs > len(y):
                bs = len(y) - i
            train_starts = torch.LongTensor(x[0][i:i+bs]).cuda()
            train_paths = torch.LongTensor(x[1][i:i+bs]).cuda()
            train_ends = torch.LongTensor(x[2][i:i+bs]).cuda()
            train_masks = torch.FloatTensor(x[3][i:i+bs]).cuda()
            train_labels = torch.LongTensor(y[i: i + bs]).cuda()

            model.train()
            optimizer.zero_grad()

            output = model(train_starts, train_paths, train_ends, train_masks)
            loss = loss_function(output, train_labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_labels)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)

        # validation
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for i in tqdm(range(0, len(y_val), BATCH_SIZE)):
            if i + bs > len(y_val):
                bs = len(y_val) - i
            val_starts = torch.LongTensor(x_val[0][i:i + bs]).cuda()
            val_paths = torch.LongTensor(x_val[1][i:i + bs]).cuda()
            val_ends = torch.LongTensor(x_val[2][i:i + bs]).cuda()
            val_masks = torch.FloatTensor(x_val[3][i:i + bs]).cuda()
            val_labels = torch.LongTensor(y_val[i: i + bs]).cuda()

            model.eval()

            output = model(val_starts, val_paths, val_ends, val_masks)
            loss = loss_function(output, val_labels)

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item() * len(val_labels)
            if total_acc/total > best_acc:
                best_model = model

            #print("loss: %f   acc: %f" % (loss.item(), (predicted == val_labels).sum().float() / len(val_labels)))

        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc / total)
        end_time = time.time()
        # scheduler.step()

        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch], train_acc_[epoch],
                 val_acc_[epoch], end_time - start_time))

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)

    # test
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    model = best_model
    for i in tqdm(range(0, len(y_test), BATCH_SIZE)):
        if i + bs > len(y_test):
            bs = len(y_test) - i
        test_starts = torch.LongTensor(x_test[0][i:i + bs]).cuda()
        test_paths = torch.LongTensor(x_test[1][i:i + bs]).cuda()
        test_ends = torch.LongTensor(x_test[2][i:i + bs]).cuda()
        test_masks = torch.FloatTensor(x_test[3][i:i + bs]).cuda()
        test_labels = torch.LongTensor(y_test[i: i + bs]).cuda()

        model.eval()

        output = model(test_starts, test_paths, test_ends, test_masks)
        loss = loss_function(output, test_labels)

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_labels)

    print("Testing results(Acc):", total_acc.item() / total)
















