from model import Code2VecCloneDetector
from process_data import C_code
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import csv
import os


if __name__ == '__main__':
    BATCH_SIZE = 64
    t = 5
    # Load Data
    data = C_code()

    print('Loading data...')

    x_test = data.load_data('test', t)
    y_test = data.get_label('test', t)
    print('test dataset: ' + str(np.array(x_test).shape))

    print('Done Data Loader')

    # Load Model
    # nodes_dim = 192697   # C: 9999  Java: 69840  java-astnn:218464
    # paths_dim = 84035     # C:183761 Java: 212200   java-astnn: 88010
    nodes_dim = 2536   # C: 9999  Java: 69840  java-astnn:218464
    paths_dim = 31628     # C:183761 Java: 212200   java-astnn: 88010
    embedding_dim = 128
    code_vector_size = 3 * embedding_dim
    dropout = 0.25
    USE_GPU = True

    model = Code2VecCloneDetector(nodes_dim+1, paths_dim+1, embedding_dim, code_vector_size, dropout)

    if USE_GPU:
        model.cuda()

    model.eval()

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.BCELoss()
    print('Done creating Code2VecModel')

    EPOCHS = 10
    categories = 5
    #writer = SummaryWriter('log/1103')
    PATH = './model/clone_model_java.pkl'
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Testing...')
        # # test
    total_loss = 0.0
    total = 0.0
    predicts = []
    trues = []
    bs = BATCH_SIZE
    precision, recall, f1 = 0, 0, 0
    for i in tqdm(range(0, len(y_test), BATCH_SIZE)):
        if i + bs > len(y_test):
            bs = len(y_test) - i
        test_starts_x1 = torch.LongTensor(x_test[0][i:i + bs]).cuda()
        test_paths_x1 = torch.LongTensor(x_test[1][i:i + bs]).cuda()
        test_ends_x1 = torch.LongTensor(x_test[2][i:i + bs]).cuda()
        test_masks_x1 = torch.FloatTensor(x_test[3][i:i + bs]).cuda()
        test_starts_x2 = torch.LongTensor(x_test[4][i:i + bs]).cuda()
        test_paths_x2 = torch.LongTensor(x_test[5][i:i + bs]).cuda()
        test_ends_x2 = torch.LongTensor(x_test[6][i:i + bs]).cuda()
        test_masks_x2 = torch.FloatTensor(x_test[7][i:i + bs]).cuda()
        test_labels = torch.FloatTensor(y_test[i: i + bs]).cuda()

        model.eval()

        output = model(test_starts_x1, test_paths_x1, test_ends_x1, test_masks_x1,
                       test_starts_x2, test_paths_x2, test_ends_x2, test_masks_x2)
        # loss = loss_function(output, test_labels)

        total += len(test_labels)
        predicted = (output.data > 0.5).cpu().numpy()
        predicts.extend(predicted)
        trues.extend(test_labels.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))

    result = pd.DataFrame(np.array(predicts), columns=['predict'])
    result['true'] = pd.DataFrame(np.array(trues))
    result['label'] = pd.DataFrame(np.array(trues))
    result.loc[result['label'] > 0, 'label'] = 1
    weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
    for k in range(1, categories + 1):
        trues_ = result[result['true'].isin([0, k])]['label'].values
        predicts_ = result[result['true'].isin([0, k])]['predict'].values
        p, r, f, _ = precision_recall_fscore_support(trues_, predicts_, average='binary')
        precision += weights[k] * p
        recall += weights[k] * r
        f1 += weights[k] * f
        print("Type-" + str(k) + ": " + str(p) + " " + str(r) + " " + str(f))
    
    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
