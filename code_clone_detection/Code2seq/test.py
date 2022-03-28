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

    test_data = pd.read_csv('./data/java/test_data.csv')
    test_data = test_data.replace(-1, 0)
    # test_data.loc[test_data['label'] > 0, 'label'] = 1

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
    checkpoint = torch.load(PATH)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])

    # print('Start training...')
    max_contexts = 500
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    predicts = []
    trues = []
    bs = BATCH_SIZE
    model.eval()
    for i in tqdm(range(0, test_num, bs)):
        if i + bs > test_num:
            bs = test_num - i

        batch_data = test_data[i:i + bs]
        test_labels = torch.FloatTensor(batch_data['label'].values.tolist()).cuda()

        test_starts_x1, test_paths_x1, test_ends_x1 = get_data(batch_data['id1'].values.tolist(), max_contexts)
        test_starts_x2, test_paths_x2, test_ends_x2 = get_data(batch_data['id2'].values.tolist(), max_contexts)

        output = model(test_starts_x1, test_paths_x1, test_ends_x1,
                       test_starts_x2, test_paths_x2, test_ends_x2)
        predicted = (output.data > 0.5).cpu().numpy()
        predicts.extend(predicted)
        trues.extend(test_labels.cpu().numpy())

    result = pd.DataFrame(np.array(predicts), columns=['predict'])
    result['true'] = pd.DataFrame(np.array(trues))
    result['label'] = pd.DataFrame(np.array(trues))
    result.loc[result['label'] > 0, 'label'] = 1
    for k in range(1, 6):
        trues_ = result[result['true'].isin([0, k])]['label'].values
        predicts_ = result[result['true'].isin([0, k])]['predict'].values
        p, r, f, _ = precision_recall_fscore_support(trues_, predicts_, average='binary')
        print("Type-" + str(k) + ": " + str(p) + " " + str(r) + " " + str(f))

















