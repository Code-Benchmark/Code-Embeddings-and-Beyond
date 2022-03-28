from model import Code2VecCloneDetector
from process_data import C_code
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os


if __name__ == '__main__':
    BATCH_SIZE = 64
    t = 5
    # Load Data
    data = C_code()

    print('Loading data...')

    x = data.load_data('train', t)
    y = data.get_label('train', t)
    print('train dataset: ' + str(np.array(x).shape))

    x_val = data.load_data('dev', t)
    y_val = data.get_label('dev', t)
    print('valid dataset: ' + str(np.array(x_val).shape))

    print('Done Data Loader')

    # Load Model
    nodes_dim = 192697   # C: 9999  Java: 69840  java-astnn:218464
    paths_dim = 84035     # C:183761 Java: 212200   java-astnn: 88010
    embedding_dim = 128
    code_vector_size = 3 * embedding_dim
    dropout = 0.25
    USE_GPU = True

    model = Code2VecCloneDetector(nodes_dim+1, paths_dim+1, embedding_dim, code_vector_size, dropout)

    if USE_GPU:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.BCELoss()
    print('Done creating Code2VecModel')

    EPOCHS = 10
    #writer = SummaryWriter('log/1103')
    PATH = './model/clone_model.pkl'
    # checkpoint = torch.load(PATH)
    # start_epoch = checkpoint['epoch']
    # model.load_state(checkpoint['model_state_dict'])

    print('Start training...')

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
        for i in tqdm(range(0, len(y), BATCH_SIZE)):
            if i+bs > len(y):
                bs = len(y) - i
            train_starts_x1 = torch.LongTensor(x[0][i:i+bs]).cuda()
            train_paths_x1 = torch.LongTensor(x[1][i:i+bs]).cuda()
            train_ends_x1 = torch.LongTensor(x[2][i:i+bs]).cuda()
            train_masks_x1 = torch.FloatTensor(x[3][i:i+bs]).cuda()

            train_starts_x2 = torch.LongTensor(x[4][i:i+bs]).cuda()
            train_paths_x2 = torch.LongTensor(x[5][i:i+bs]).cuda()
            train_ends_x2 = torch.LongTensor(x[6][i:i+bs]).cuda()
            train_masks_x2 = torch.FloatTensor(x[7][i:i+bs]).cuda()

            train_labels = torch.FloatTensor(y[i: i + bs]).cuda()

            model.train()
            optimizer.zero_grad()

            output = model(train_starts_x1, train_paths_x1, train_ends_x1, train_masks_x1,
                           train_starts_x2, train_paths_x2, train_ends_x2, train_masks_x2)

            # loss = loss_function(output, train_labels)

            loss.backward()
            optimizer.step()

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
        for i in tqdm(range(0, len(y_val), BATCH_SIZE)):
            if i + bs > len(y_val):
                bs = len(y_val) - i
            val_starts_x1 = torch.LongTensor(x_val[0][i:i + bs]).cuda()
            val_paths_x1 = torch.LongTensor(x_val[1][i:i + bs]).cuda()
            val_ends_x1 = torch.LongTensor(x_val[2][i:i + bs]).cuda()
            val_masks_x1 = torch.FloatTensor(x_val[3][i:i + bs]).cuda()
            val_starts_x2 = torch.LongTensor(x_val[4][i:i + bs]).cuda()
            val_paths_x2 = torch.LongTensor(x_val[5][i:i + bs]).cuda()
            val_ends_x2 = torch.LongTensor(x_val[6][i:i + bs]).cuda()
            val_masks_x2 = torch.FloatTensor(x_val[7][i:i + bs]).cuda()
            val_labels = torch.FloatTensor(y_val[i: i + bs]).cuda()


            model.eval()

            output = model(val_starts_x1, val_paths_x1, val_ends_x1, val_masks_x1,
                           val_starts_x2, val_paths_x2, val_ends_x2, val_masks_x2 )
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
    
