import pandas as pd
import random
import torch
import pickle
import time
import numpy as np
from model import TBCNN
import math
import argparse
import load_javadata
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import sampling
from sklearn.metrics import precision_recall_fscore_support


logdir = ''
infile = ''
modelpath = r'clonedata.pt'
conv_feature = 600
batch_size = 1
epochs = 1
checkpoint_every = ''
USE_GPU = 1


def truncated_normal_(tensor, mean=0, std=0.09):  # https://zhuanlan.zhihu.com/p/83609874  tf.trunc_normal()

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def load_data(rootfile, treefile, embeddingfile):
    trees_file = rootfile + treefile

    with open(trees_file, 'rb') as fh:
        trees = pickle.load(fh)

    # id = list(set(id))
    embedding_file = rootfile + embeddingfile
    with open(embedding_file, 'rb') as fh:
        embeddings,lookup = pickle.load(fh)
        num_feats = len(embeddings[0])


    return trees, embeddings, lookup, num_feats



def train(args):
    """train data"""
    lang = args.lang
    categories = 1

    if lang == 'java':
        categories = 5
    print("Train for ", str.upper(lang))


    print("loading data...")
    trainpair = pd.read_csv(args.path + '/train_example.csv')

    devpair = pd.read_csv(args.path + '/dev_idpairs.csv')
    testpair = pd.read_csv(args.path + '/test_idpairs.csv')


    code = pd.read_csv(args.path + '/codes.csv')

    traindata = load_javadata.merge(trainpair,code)
    devdata = load_javadata.merge(devpair,code)
    testdata = load_javadata.merge(testpair,code)

    trees, embeddings, lookup, num_feats = load_data(args.path, '/example_trees.pkl','/example_embedding.pkl')



    w_t = truncated_normal_(torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)).cuda()
    w_l = truncated_normal_(torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)).cuda()
    w_r = truncated_normal_(torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)).cuda()
    init = truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / num_feats))
    b_conv = torch.tensor(init).cuda()
    ##!!!
    w_h = truncated_normal_(torch.zeros((conv_feature, 1)), std=1.0 / math.sqrt(conv_feature)).cuda()
    b_h = truncated_normal_(torch.zeros(1, ), std=math.sqrt(2.0 / conv_feature)).cuda()

    p1 = torch.tensor(num_feats).cuda()
    # p2 = torch.tensor(len(id)).cuda()
    p2 = torch.tensor(1).cuda()

    p3 = torch.tensor(conv_feature).cuda()

    model = TBCNN(p1, p2, p3, w_t, w_l, w_r, b_conv, w_h, b_h)

    if USE_GPU:
        model.cuda()

    parameter = model.parameters()
    optimizer = torch.optim.Adamax(parameter)
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.BCELoss()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    print('Start training...')

    allnodes, allchildren = sampling.gen_samples(trees, embeddings, lookup)
    t = 5
    # gen = (trees, embeddings, lookup)
    # best_model = model
    # modeldata = torch.load(modelpath)
    # model.load_state_dict(modeldata['model_state_dict'])

    for epoch in range(epochs):
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        bs = batch_size
        predicts = []
        trues = []

        for i in tqdm(range(0,len(traindata),bs)):

            if i + bs > len(traindata):
                bs = len(traindata) - i

            batch = load_javadata.get_batch(allnodes, allchildren,i,batch_size,traindata)


            nodes1,nodes2,children1,children2,batch_labels= batch

            if not nodes1:
                continue  # don't try to train on an empty batch
            if USE_GPU:
                nodes1 = torch.tensor(nodes1).cuda()
                children1 = torch.tensor(children1).cuda()

                nodes2 = torch.tensor(nodes2).cuda()
                children2 = torch.tensor(children2).cuda()
                batch_labels = torch.tensor(batch_labels).cuda()

            model.zero_grad()
            model.batch_size = len(batch_labels)
            output = model(nodes1,children1,nodes2,children2)
            # print(i)
            # loss = loss_function(output, batch_labels)

            loss = loss_function(output.cpu(), (batch_labels.cpu()).float())
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print("epoch:", epoch+1, "loss", loss.item())

            total += len(batch_labels)
            total_loss += loss.item() * len(batch_labels)
            predicted = (output.data > 0.5).cpu().numpy()
            # total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
            predicts.extend(predicted)
            trues.extend(batch_labels.cpu().numpy())



        train_loss.append(total_loss / total)
        p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        np.save('trainresult.npy',f)

        i = 0
        total_loss = 0.0
        bs = batch_size
        total = 0.0
        predicts = []
        trues = []
        for i in tqdm(range(0,len(devdata),bs)):

            if i + bs > len(devdata):
                bs = len(devdata)

            batch = sampling.get_batch(allnodes, allchildren,i,batch_size,devdata)


            dev_nodes1,dev_nodes2,dev_children1,dev_children2,dev_batch_labels= batch

            if USE_GPU:
                dev_nodes1 = torch.tensor(dev_nodes1).cuda()
                dev_children1 = torch.tensor(dev_children1).cuda()

                dev_nodes2 = torch.tensor(dev_nodes2).cuda()
                dev_children2 = torch.tensor(dev_children2).cuda()
                dev_batch_labels = torch.tensor(dev_batch_labels).cuda()

            model.batch_size = len(dev_batch_labels)
            output = model(dev_nodes1, dev_children1, dev_nodes2, dev_children2)
            loss = loss_function(output.cpu(), (dev_batch_labels.cpu()).float())
            total += len(dev_batch_labels)
            total_loss += loss.item() * len(dev_batch_labels)
            predicted = (output.data > 0.5).cpu().numpy()
            # total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
            predicts.extend(predicted)
            trues.extend(dev_batch_labels.cpu().numpy())

        val_loss.append(total_loss / total)
        p_, r_, f_, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        np.save('devresult.npy',f_)
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              % (epoch + 1, epochs, train_loss[epoch], val_loss[epoch]))
        print("Train results(P,R,F1):%.3f, %.3f, %.3f" % (p, r, f))
        print("Dev results(P,R,F1):%.3f, %.3f, %.3f" % (p_, r_, f_))

        torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, modelpath)

    
    print("testing...")
    i = 0
    total_loss = 0.0
    bs = batch_size
    total = 0.0
    predicts = []
    trues = []
    p,r,f = 0,0,0
    while i < len(testdata):

        batch = sampling.get_batch(allnodes, allchildren, i, batch_size, testdata)
        i += batch_size
        test_nodes1, test_nodes2, test_children1, test_children2, test_batch_labels = batch

        if USE_GPU:
            test_nodes1 = torch.tensor(test_nodes1).cuda()
            test_children1 = torch.tensor(test_children1).cuda()

            test_nodes2 = torch.tensor(test_nodes2).cuda()
            test_children2 = torch.tensor(test_children2).cuda()
            test_batch_labels = torch.tensor(test_batch_labels).cuda()

        model.batch_size = len(test_batch_labels)
        output = model(test_nodes1, test_children1, test_nodes2, test_children2)


        predicted = (output.data > 0.5).cpu().numpy()
        # total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
        predicts.extend(predicted)
        trues.extend(test_batch_labels.cpu().numpy())
        if i%1000 ==0:
            print("test:",i)
    if lang =='java':


        weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]

        pt, rt, ft, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        np.save('testresult.npy', pt, rt, ft)
        p += weights[t] * pt
        r += weights[t] * rt
        f += weights[t] * ft
        print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))

    elif lang =='c':
        p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (pt, rt, ft))
    np.save('trues.npy', trues)
    np.save('predicts.npy', predicts)

if __name__ == '__main__':
    filepath = ''

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument('--lang')
    # parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument('--path')
    args = parser.parse_args()

    root = r'../data'
    args.lang = 'java'
    args.path = r'../data'


    if not args.lang:
        print("No specified dataset")
        exit(1)


    # split_data('3:1:1',rootfile,filepath)

    train(args)