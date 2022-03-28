import pandas as pd
import random
import torch
import pickle
import time
from Clone.TreeCaps.clone.model import Treecaps
import Clone.TreeCaps.clone.load_javadata as load_javadata
from tqdm import tqdm
import Clone.TreeCaps.clone.sampling as sampling
from sklearn.metrics import precision_recall_fscore_support
cuda1 = torch.device('cuda:0')

logdir = ''
infile = ''
modelpath = r'clonedata.pt'
conv_feature = 64
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

def load_data(rootfile, treefile, lookupfile):
    trees_file = rootfile + treefile

    with open(trees_file, 'rb') as fh:
        trees = pickle.load(fh)

    # id = list(set(id))
    embedding_file = rootfile + lookupfile
    NODE_LIST = pd.read_pickle(embedding_file)
    dict =[]
    for i in range(len(NODE_LIST)):
        if NODE_LIST[i][1] >=5:
            dict.append(NODE_LIST[i][0] )
    dict.append('UNK')
    NODE_MAP = {x: i for (i, x) in enumerate(dict)}
    return trees, NODE_MAP



def train(args,trainpair,devpair,testpair):
    """train data"""
    lang = 'java'
    categories = 1

    if lang == 'java':
        categories = 5
    print("Train for ", str.upper(lang))


    print("loading data...")

    code = pd.read_csv(args.dataset_directory + '/codes.csv')

    traindata = load_javadata.merge(trainpair,code)
    devdata = load_javadata.merge(devpair,code)
    testdata = load_javadata.merge(testpair,code)

    trees, lookup = load_data(args.dataset_directory, '/example_trees.pkl','/lookup.pkl')

    num_feats = 128
    p1 = torch.tensor(num_feats).cuda()
    p2 = torch.tensor(1).cuda()
    p3 = torch.tensor(conv_feature).cuda()

    tlen = 104
    conv_size = 64
    num_conv = 8
    num_outputs = 1
    num_dims = 16
    cuda1 = torch.device('cuda:0')
    # train_num_feats = 128

    Wemd = torch.nn.init.uniform(torch.zeros((len(lookup), num_feats), dtype=torch.float32, device=cuda1))
    model = Treecaps(len(lookup),p1, p2, p3,conv_size, num_conv, num_feats, num_dims, num_outputs, tlen, Wemd)

    if USE_GPU:
        model.cuda()

    parameter = model.parameters()
    optimizer = torch.optim.Adamax(parameter)
    loss_function = torch.nn.BCELoss()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    print('Start training...')

    allnodes, allchildren = sampling.gen_samples(trees,lookup)
    t = 5

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
                nodes1 = torch.tensor(nodes1, device=cuda1)
                children1 = torch.tensor(children1, device=cuda1)

                nodes2 = torch.tensor(nodes2, device=cuda1)
                children2 = torch.tensor(children2, device=cuda1)
                batch_labels = torch.tensor(batch_labels, device=cuda1)

            model.zero_grad()
            model.batch_size = len(batch_labels)
            output = model(nodes1,children1,nodes2,children2)
            # print(i)
            # loss = loss_function(output, batch_labels)

            loss = loss_function(output.cpu(), (batch_labels.cpu()).float())
            loss.backward()
            optimizer.step()

            total += len(batch_labels)
            total_loss += loss.item() * len(batch_labels)
            predicted = (output.data > 0.5).cpu().numpy()
            # total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
            predicts.extend(predicted)
            trues.extend(batch_labels.cpu().numpy())

            if i % 2000 == 0:
                p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
                print("epoch:", epoch+1, "loss", loss.item(),"F:",f)


        train_loss.append(total_loss / total)
        p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')

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
                dev_nodes1 = torch.tensor(dev_nodes1, device=cuda1)
                dev_children1 = torch.tensor(dev_children1, device=cuda1)

                dev_nodes2 = torch.tensor(dev_nodes2, device=cuda1)
                dev_children2 = torch.tensor(dev_children2, device=cuda1)
                dev_batch_labels = torch.tensor(dev_batch_labels, device=cuda1)

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
        print('[Epoch: %3d/%3d] Training Loss: %.4f'
              % (epoch + 1, epochs, train_loss[epoch]))
        print("Train results(P,R,F1):%.3f, %.3f, %.3f" % (p, r, f))
        torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, args.model_path)

    
    print("testing...")
    i = 0
    total_loss = 0.0
    bs = batch_size
    total = 0.0
    predicts = []
    trues = []
    p,r,f = 0,0,0
    for i in tqdm(range(0, len(testdata), batch_size)):
        batch = sampling.get_batch(allnodes, allchildren, i, batch_size, testdata)

        test_nodes1, test_nodes2, test_children1, test_children2, test_batch_labels = batch

        if USE_GPU:
            test_nodes1 = torch.tensor(test_nodes1, device=cuda1)
            test_children1 = torch.tensor(test_children1, device=cuda1)

            test_nodes2 = torch.tensor(test_nodes2, device=cuda1)
            test_children2 = torch.tensor(test_children2, device=cuda1)
            test_batch_labels = torch.tensor(test_batch_labels, device=cuda1)

        model.batch_size = len(test_batch_labels)
        output = model(test_nodes1, test_children1, test_nodes2, test_children2)


        predicted = (output.data > 0.5).cpu().numpy()
        # total_acc += (precicted == torch.tensor(batch_labels).cuda()).sum()
        predicts.extend(predicted)
        trues.extend(test_batch_labels.cpu().numpy())

    if lang =='java':


        weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]

        pt, rt, ft, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        p += weights[t] * pt
        r += weights[t] * rt
        f += weights[t] * ft
        print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))

    elif lang =='c':
        p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (pt, rt, ft))