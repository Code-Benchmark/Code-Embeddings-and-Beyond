import pandas as pd
import torch
from tqdm import tqdm
import time
from models import GGNNCloneDetector
from sklearn.metrics import precision_recall_fscore_support

import json

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(eval(item['code_ids']))
        labels.append(item['label']-1)
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    root = 'data/'
    lang = 'java'   # java
    train_data = pd.read_csv(root + lang + '/train_idpairs.csv')
    train_data = train_data.replace(-1, 0)
    val_data = pd.read_csv(root + lang + '/dev_idpairs.csv')
    val_data = val_data.replace(-1, 0)
    test_data = pd.read_csv(root + lang + '/test_idpairs.csv')
    test_data = test_data.replace(-1, 0)
    test_data.loc[test_data['label'] > 0, 'label'] = 1

    data_json = json.loads(open(root+lang+'/ggnn.json', 'r').read())

    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 64
    USE_GPU = True
    vocablen = 77410   # java: 77410
    edge_type = 7
    EMBEDDING_DIM = 200
    num_layers = 4
    device = torch.device('cuda:0')
    model = GGNNCloneDetector(vocablen, EMBEDDING_DIM, num_layers, device)
    PATH = './model/model_clone_java.pkl'
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        # bs = BATCH_SIZE
        predicts = []
        trues = []
        for i in tqdm(range(0, len(train_data))): 
            train_label = train_data[i:i+1]['label'].values[0]
            train_label_c = torch.FloatTensor([train_label]).cuda()
            train_input_1 = data_json[str(train_data[i:i+1]['id1'].values[0])]
            train_input_2 = data_json[str(train_data[i:i + 1]['id2'].values[0])]
            model.zero_grad()
            model.batch_size = 1
            output = model(train_input_1, train_input_2)

            loss = loss_function(output, train_label_c)
            loss.backward()
            optimizer.step()

            total += 1
            total_loss += loss.item()
            predicted = (output.data > 0.5).int().cpu().numpy()
            predicts.extend(predicted)
            trues.append(train_label)

        train_loss_.append(total_loss / total)
        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        bs = BATCH_SIZE
        predicts = []
        trues = []
        for i in tqdm(range(0, len(val_data))): #len(val_data)
            val_label = val_data[i:i + 1]['label'].values[0]
            val_label_c = torch.FloatTensor([val_label]).cuda()
            val_input_1 = data_json[str(val_data[i:i + 1]['id1'].values[0])]
            val_input_2 = data_json[str(val_data[i:i + 1]['id2'].values[0])]

            model.batch_size = 1
            output = model(val_input_1, val_input_2)

            loss = loss_function(output, val_label_c)

            total += 1
            total_loss += loss.item()
            predicted = (output.data > 0.5).int().cpu().numpy()
            predicts.extend(predicted)
            trues.append(val_label)

        val_loss_.append(total_loss / total)
        precision_, recall_, f1_, _ = precision_recall_fscore_support(trues, predicts, average='binary')

        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch]))
        print("Train results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
        print("Dev results(P,R,F1):%.3f, %.3f, %.3f" % (precision_, recall_, f1_))

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    predicts = []
    trues = []
    for i in tqdm(range(0, len(test_data))):
        test_label = val_data[i:i + 1]['label'].values[0]
        test_label_c = torch.FloatTensor([test_label]).cuda()
        test_input_1 = data_json[str(test_data[i:i + 1]['id1'].values[0])]
        test_input_2 = data_json[str(test_data[i:i + 1]['id2'].values[0])]
        #model.hidden = model.init_hidden()
        output = model(test_input_1, test_input_2)

        loss = loss_function(output, test_label_c)

        total += 1
        total_loss += loss.item()
        predicted = (output.data > 0.5).int().cpu().numpy()
        predicts.extend(predicted)
        trues.append(val_label)

    precision_, recall_, f1_, _ = precision_recall_fscore_support(trues, predicts, average='binary')
    print("Test results(P,R,F1):%.3f, %.3f, %.3f" % (precision_, recall_, f1_))
