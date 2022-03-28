from model import Code2Seq
from process_data import C_code
import torch
import time
import random
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

    x = data.load_data_2('train')
    y = data.get_label('train')
    print('train dataset: ' + str(np.array(x).shape))

    x_val = data.load_data_2('dev')
    y_val = data.get_label('dev')
    print('valid dataset: ' + str(np.array(x_val).shape))

    x_test = data.load_data_2('test')
    y_test = data.get_label('test')
    print('test dataset: ' + str(np.array(x_test).shape))
    #
    print('Done Data Loader')

    # Load Model
    nodes_dim = 6851
    paths_dim = 168
    embedding_dim = 128
    code_vector_size = 4 * embedding_dim
    output_dim = 104
    dropout = 0.25  # 0.25
    USE_GPU = True

    model = Code2Seq(nodes_dim+1, paths_dim+1, embedding_dim, code_vector_size, output_dim, dropout)

    if USE_GPU:
        model.cuda()
    decay_ratio = 0.95
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()
    print('Done creating Code2SeqModel')

    EPOCHS = 300
    max_contexts = 200
    writer = SummaryWriter('log')
    PATH = './model/c2s_clf_model_2'
    # checkpoint = torch.load(PATH)
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['model_state_dict'])

    x_code_test_true = []
    for sample in x_test:
        if len(sample) > max_contexts:
            sample = random.sample(sample, max_contexts)
        else:
            sample = sample[0:max_contexts]
        x_code_test_true.append(sample)
    x_code_test_true = np.array(x_code_test_true).swapaxes(1, 2)

    print('Start training...')
    cnt = 0
    cnt_v = 0
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
        model.train()
        bs = BATCH_SIZE
        for i in range(0, len(y), bs):
            if i+bs > len(y):
                bs = len(y) - i
            x_code_true = []
            for sample in x[i:i+bs]:
                if len(sample) > max_contexts:
                    sample = random.sample(sample, max_contexts)
                else:
                    sample = sample[0:max_contexts]
                x_code_true.append(sample)
            x_code_true = np.array(x_code_true).swapaxes(1, 2)
            train_starts = torch.LongTensor(x_code_true[:,0,:].tolist()).cuda()
            train_paths = torch.LongTensor(x_code_true[:,1,:].tolist()).cuda()
            train_ends = torch.LongTensor(x_code_true[:,2,:].tolist()).cuda()
            train_masks = torch.FloatTensor(x_code_true[:,3,:].tolist()).cuda()
            train_labels = torch.LongTensor(y[i: i + bs]).cuda()

            optimizer.zero_grad()

            output = model(train_starts, train_paths, train_ends,train_masks)
            loss = loss_function(output, train_labels)

            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', loss.item(), cnt)

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_labels)
            writer.add_scalar('train/acc', total_acc / total, cnt)
            cnt = cnt + 1

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)

        #validation
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        model.eval()
        bs = BATCH_SIZE
        for i in range(0, len(y_val), bs):
            if i + bs > len(y_val):
                bs = len(y_val) - i
            x_code_val_true = []
            for sample in x_val[i:i + bs]:
                if len(sample) > max_contexts:
                    sample = random.sample(sample, max_contexts)
                else:
                    sample = sample[0:max_contexts]
                x_code_val_true.append(sample)
            x_code_val_true = np.array(x_code_val_true).swapaxes(1, 2)
            val_starts = torch.LongTensor(x_code_val_true[:, 0, :].tolist()).cuda()
            val_paths = torch.LongTensor(x_code_val_true[:, 1, :].tolist()).cuda()
            val_ends = torch.LongTensor(x_code_val_true[:, 2, :].tolist()).cuda()
            val_masks = torch.FloatTensor(x_code_val_true[:, 3, :].tolist()).cuda()
            val_labels = torch.LongTensor(y_val[i: i + bs]).cuda()

            output = model(val_starts, val_paths, val_ends,val_masks)
            loss = loss_function(output, val_labels)

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item() * len(val_labels)

            if total_acc/total > best_acc:
                best_model = model
            
            writer.add_scalar('validation/loss', loss.item(), cnt_v)
            writer.add_scalar('validation/acc', total_acc / total, cnt_v)
            cnt_v = cnt_v + 1

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
    model.eval()
    bs = BATCH_SIZE
    model = best_model
    with torch.no_grad():
        for i in range(0, len(y_test), bs):
            if i + bs > len(y_test):
                bs = len(y_test) - i
            test_starts = torch.LongTensor(x_code_test_true[i:i + bs, 0, :].tolist()).cuda()
            test_paths = torch.LongTensor(x_code_test_true[i:i + bs, 1, :].tolist()).cuda()
            test_ends = torch.LongTensor(x_code_test_true[i:i + bs, 2, :].tolist()).cuda()
            test_masks = torch.FloatTensor(x_code_test_true[i:i + bs, 3, :].tolist()).cuda()
            test_labels = torch.LongTensor(y_test[i: i + bs]).cuda()

            output = model(test_starts, test_paths, test_ends,test_masks)
            # loss = loss_function(output, test_labels)

            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            # total_loss += loss.item() * len(test_labels)

    print("Testing results(Acc):", total_acc.item() / total)
















