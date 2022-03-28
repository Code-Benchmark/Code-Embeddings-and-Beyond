import pandas as pd
import random
import torch
import pickle
import time
import numpy as np
import TBCNN.classifier.network as network
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import TBCNN.classifier.sampling as sampling
# from tensorboardX import SummaryWriter

def truncated_normal_(tensor, mean=0, std=0.09):  # https://zhuanlan.zhihu.com/p/83609874  tf.trunc_normal()

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def load_data(rootfile, treefile, embeddingfile,type):
    trees_file = rootfile + treefile
    if type == 'train':
        with open(trees_file, 'rb') as fh:
            trees, _,_, label = pickle.load(fh)
    elif type == 'dev':
        with open(trees_file, 'rb') as fh:
            _, trees,_, label = pickle.load(fh)
    elif type == 'test':
        with open(trees_file, 'rb') as fh:
            _, _,trees, label = pickle.load(fh)
    label = list(set(label))
    embedding_file = rootfile + embeddingfile
    with open(embedding_file, 'rb') as fh:
        embeddings,lookup = pickle.load(fh)
        num_feats = len(embeddings[0])
    return trees, label, embeddings, lookup, num_feats

def train(args,train_data,dev_data,test_data):
    epochs = args.epochs
    batch_size = args.batch_size
    model_path = args.model_path
    lr = args.lr
    cuda = args.cuda
    USE_GPU = args.USE_GPU
    train_trees, train_label, train_embeddings, train_lookup, train_num_feats = train_data
    dev_trees, dev_label, dev_embeddings, dev_lookup, dev_num_feats = dev_data
    test_trees, test_label, test_embeddings, test_lookup, test_num_feats = test_data

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(train_trees)
    random.seed(randnum)
    random.shuffle(train_label)
    tlen = 104
    conv_feature = 600

    w_t = truncated_normal_(torch.zeros((train_num_feats, conv_feature)), std=1.0 / math.sqrt(train_num_feats)).cuda()
    w_l = truncated_normal_(torch.zeros((train_num_feats, conv_feature)), std=1.0 / math.sqrt(train_num_feats)).cuda()
    w_r = truncated_normal_(torch.zeros((train_num_feats, conv_feature)), std=1.0 / math.sqrt(train_num_feats)).cuda()
    init = truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / train_num_feats))
    b_conv = torch.tensor(init).cuda()
    w_h = truncated_normal_(torch.zeros((conv_feature, tlen)), std=1.0 / math.sqrt(conv_feature)).cuda()
    b_h = truncated_normal_(torch.zeros(tlen, ), std=math.sqrt(2.0 / conv_feature)).cuda()
    p1 = torch.tensor(train_num_feats).cuda()
    p2 = torch.tensor(tlen).cuda()
    p3 = torch.tensor(conv_feature).cuda()

    model = network.TBCNN(p1, p2, p3, w_t, w_l, w_r, b_conv, w_h, b_h)

    if USE_GPU:
        model.cuda()

    parameter = model.parameters()
    optimizer = torch.optim.Adamax(parameter,lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    print('Start training...')
    j = 0
    best_model = model
    for epoch in range(epochs):
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        for i, batch in enumerate(
                sampling.batch_samples(sampling.gen_samples(train_trees, train_label, train_embeddings, train_lookup),
                                       batch_size)):


            nodes, children, batch_labels = batch

            if not nodes:
                continue  # don't try to train on an empty batch

            nodes = torch.tensor(nodes).cuda()
            children = torch.tensor(children).cuda()
            batch_labels = torch.tensor(batch_labels).cuda()
            nodes.requires_grad=True
            nodes.retain_grad()
            model.zero_grad()
            model.batch_size = len(batch_labels)
            output = model(nodes, children)


            loss = loss_function(output, batch_labels)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print("epoch:", epoch+1, "loss", loss.item())

            j = j + 1
            _, precicted = torch.max(output.data, 1)
            total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
            total += len(batch_labels)
            total_loss += loss.item() * len(batch_labels)

        train_loss.append(total_loss / total)
        train_acc.append(total_acc.item() / total)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, model_path)

        print(train_acc)
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0

        i = 0
        for i, batch in enumerate(
                sampling.batch_samples(sampling.gen_samples(dev_trees, dev_label, dev_embeddings, dev_lookup),
                                       batch_size)):
            dev_nodes, dev_children, batch_labels = batch
            if USE_GPU:
                dev_nodes = (torch.tensor(dev_nodes)).cuda()
                dev_children = (torch.tensor(dev_children)).cuda()
                batch_labels = (torch.tensor(batch_labels)).cuda()

            model.batch_size = len(batch_labels)
            output = model(dev_nodes, dev_children)

            loss = loss_function(output, batch_labels)


            _, precicted = torch.max(output.data, 1)
            total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
            total += len(batch_labels)
            total_loss += loss.item() * len(batch_labels)


        val_loss.append(total_loss / total)
        val_acc.append(total_acc.item() / total)

        end_time = time.time()

        if (total_acc / total).item() > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, epochs, train_loss[epoch], val_loss[epoch],
                 train_acc[epoch], val_acc[epoch], end_time - start_time))
    
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0

    i = 0
    model = best_model

    for i, batch in enumerate(
            sampling.batch_samples(sampling.gen_samples(test_trees, test_label, test_embeddings, test_lookup),
                                   batch_size)):
        test_nodes, test_children, batch_labels = batch

        if USE_GPU:
            test_nodes = (torch.tensor(test_nodes)).cuda()
            test_children = (torch.tensor(test_children)).cuda()
            batch_labels = (torch.tensor(batch_labels)).cuda()

        # if not nodes:
        #     continue
            # don't try to train on an empty batch

        model.batch_size = len(batch_labels)
        output = model(test_nodes, test_children)

        loss = loss_function(output, batch_labels)

        _, precicted = torch.max(output.data, 1)
        total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
        total += len(batch_labels)
        total_loss += loss.item() * len(batch_labels)

    print("Testing result(ACC):", total_acc.item() / total)
