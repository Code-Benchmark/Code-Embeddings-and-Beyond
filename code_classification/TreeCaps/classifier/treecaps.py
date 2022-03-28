import pandas as pd
import random
import torch
import pickle
import time
import numpy as np
from TreeCaps.classifier.network import Treecaps
import TreeCaps.classifier.network as network
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import TreeCaps.classifier.sampling as sampling
from tqdm import tqdm

def truncated_normal_(tensor, mean=0, std=0.09):  # https://zhuanlan.zhihu.com/p/83609874  tf.trunc_normal()

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]

def train(args,train_data,dev_data,test_data):
    """train data"""
    epochs = args.epochs
    conv_feature = 64
    train_trees, train_label, train_lookup = train_data
    dev_trees, dev_label, dev_lookup= dev_data
    test_trees, test_label, test_lookup= test_data

    randnum = 66
    random.seed(randnum)
    random.shuffle(train_trees)
    random.seed(randnum)
    random.shuffle(train_label)
    tlen = 104
    conv_size = 64
    num_conv = 8
    num_outputs = 104
    num_dims = 16
    cuda1 = torch.device('cuda:'+args.cuda)
    num_feats = 128
    batch_size = args.batch_size
    p1 = torch.as_tensor(num_feats, device=cuda1)
    p2 = torch.as_tensor(tlen, device=cuda1)
    p3 = torch.as_tensor(conv_feature, device=cuda1)
    Wemd = torch.nn.init.uniform(torch.zeros((len(train_lookup), num_feats), dtype=torch.float32, device=cuda1))
    model = Treecaps(len(train_lookup),p1, p2, p3,conv_size, num_conv, num_feats, num_dims, num_outputs, tlen, Wemd)

    if args.USE_GPU:
        model.cuda()

    parameter = model.parameters()
    optimizer = torch.optim.RAdam(parameter, lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    print('Start training...')
    j = 0

    best_model = model
    allnode, allchildren, alllab, alllabel = sampling.gen_allsamples(train_trees, train_label,train_lookup)
    devnodes, devchildren, devlab, devlabel = sampling.gen_allsamples(dev_trees, dev_label, dev_lookup)
    testnodes, testchildren, testlab, testlabel = sampling.gen_allsamples(test_trees, test_label, test_lookup)
    for epoch in range(epochs):
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        for i in tqdm(range(len(train_trees))):

            nodes, children, lab, batch_labels = sampling.batch_onesample(allnode, allchildren, alllab, alllabel, i,
                                                                          batch_size)
            if not nodes:
                continue  # don't try to train on an empty batch

            nodes = torch.as_tensor(nodes, device=cuda1)
            children = torch.as_tensor(children, device=cuda1)
            # with torch.autograd.profiler.profile(with_stack=True, profile_memory=True,use_cuda=True) as prof:
            batch_labels = torch.as_tensor(batch_labels,device=cuda1)
            model.zero_grad()
            model.batch_size = len(batch_labels)

            output = model(nodes, children)
            # _, loss = loss_function(output, torch.tensor(lab, device=cuda1))
            loss = loss_function(output, batch_labels)
            loss.backward()
            optimizer.step()
            # print(prof)

            if i % 200 == 0:
                end_time_s = time.time()
                print("epoch:", epoch+1, "loss", loss.item(),'total_acc:',total_acc,'total:',total,'time:',end_time_s-start_time)


            j = j + 1
            _, precicted = torch.max(output[0].data,dim=0)
            total_acc += (precicted == torch.as_tensor(batch_labels,device=cuda1)).sum()
            total += len(batch_labels)
            total_loss += loss.item() * len(batch_labels)

        train_loss.append(total_loss / total)
        train_acc.append(total_acc.item() / total)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, args.model_path)

        print(train_acc)
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0


        for i in tqdm(range(len(dev_trees))):
            dev_nodes, dev_children, lab,batch_labels = sampling.batch_onesample(devnodes, devchildren, devlab, devlabel,i,
                                                                          batch_size)
            if args.USE_GPU:
                dev_nodes = (torch.as_tensor(dev_nodes, device=cuda1))
                dev_children = (torch.as_tensor(dev_children, device=cuda1))
                batch_labels = (torch.as_tensor(batch_labels, device=cuda1))

            # bsl = torch.zeros(batch_size, 104).cuda().scatter_(1, batch_labels.unsqueeze(1).cuda(), 1)
            model.batch_size = len(batch_labels)
            output = model(dev_nodes, dev_children)

            # _,loss = loss_function(output,torch.tensor(lab, device=cuda1))
            loss = loss_function(output, batch_labels)

            _, precicted = torch.max(output.data, 1)
            total_acc += (precicted == torch.as_tensor(batch_labels, device=cuda1)).sum()
            total += len(batch_labels)
            total_loss += loss.item() * len(batch_labels)


        val_loss.append(total_loss / total)
        val_acc.append(total_acc.item() / total)

        end_time = time.time()

        if (total_acc / total).item() > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, args.epochs, train_loss[epoch], val_loss[epoch],
                 train_acc[epoch], val_acc[epoch], end_time - start_time))
    
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0

    i = 0
    model = best_model

    for i in tqdm(range(len(test_trees))):
        test_nodes, test_children, lab, batch_labels = sampling.batch_onesample(testnodes, testchildren, testlab, testlabel,i,
                                                                          batch_size)
        # bsl = torch.zeros(batch_size, 104).cuda().scatter_(1, batch_labels.unsqueeze(1).cuda(), 1)
        if args.USE_GPU:
            test_nodes = (torch.as_tensor(test_nodes, device=cuda1))
            test_children = (torch.as_tensor(test_children,device=cuda1))
            batch_labels = (torch.as_tensor(batch_labels,device=cuda1))

        model.batch_size = len(batch_labels)
        output = model(test_nodes, test_children)

        # _,loss = loss_function(output, torch.tensor(lab, device=cuda1))
        loss = loss_function(output, batch_labels)

        _, precicted = torch.max(output.data, 1)
        total_acc += (precicted == torch.as_tensor(batch_labels,device=cuda1)).sum()
        total += len(batch_labels)
        total_loss += loss.item() * len(batch_labels)
    print("Testing result(ACC):", total_acc.item() / total)
