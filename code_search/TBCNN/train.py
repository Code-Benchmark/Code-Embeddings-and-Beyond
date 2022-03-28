import pandas as pd
import torch
import time
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from model.model import SearchModel
from model.loss import RankLoss
import random

from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import pickle
import math
import sampling
import load_javadata

warnings.filterwarnings('ignore')


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2 = [], []
    fp_x1, fp_x2 = [], []

    for _, item in tmp.iterrows():
        x1.append(eval(item['code_ids_']))
        x2.append(eval(item['doc_ids_']))
        fp_x1.append(eval(item['code_ids_']))
        fp_x2.append(eval(item['fp_doc_ids_']))
        # labels.append([item['label']])
    return x1, x2, fp_x1, fp_x2  # , torch.FloatTensor(labels)


def ACC(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1
    return sum / float(len(real))


def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


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
        embeddings, lookup = pickle.load(fh)
        num_feats = len(embeddings[0])
    return trees, embeddings, lookup, num_feats


if __name__ == '__main__':

    root = r'data'
    modelpath = r'tbcnn_codesearch_0126'
    modelpt = r'tbcnn_codesearch_0126'
    lang = 'java'
    # conv_feature = 600

    print("Train for ", str.upper(lang))


    word2vec = Word2Vec.load("data/doc_token_w2v_200").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    EPOCHS = 1
    VALID_BATCH_SIZE = 999
    BATCH_SIZE = 1
    USE_GPU = True
    conv_feature = 600
    feature_size = 30
    code_model = "Transformer"

    train_csv, test_csv, all_csv,train, test = load_javadata.read_train_data(root + '/example.pkl', root + '/example.csv')
    print("test_csv:",len(test_csv))
    print("test:",len(test))
    ###
    #test_csv = train_csv
    #test = train

    # tbcnn---
    trees, tbembeddings, lookup, num_feats = load_data(root, '/example_trees.pkl', '/example_embeddings.pkl')

    w_t = truncated_normal_(torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)).cuda()
    w_l = truncated_normal_(torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)).cuda()
    w_r = truncated_normal_(torch.zeros((num_feats, conv_feature)), std=1.0 / math.sqrt(num_feats)).cuda()
    init = truncated_normal_(torch.zeros(conv_feature, ), std=math.sqrt(2.0 / num_feats))
    b_conv = torch.tensor(init).cuda()
    ##!!!
    w_h = truncated_normal_(torch.zeros((conv_feature, 200)), std=1.0 / math.sqrt(conv_feature)).cuda()
    b_h = truncated_normal_(torch.zeros(200, ), std=math.sqrt(2.0 / conv_feature)).cuda()
    feature_size = torch.tensor(feature_size).cuda()
    conv_feature = torch.tensor(conv_feature).cuda()
    model = SearchModel(code_model, EMBEDDING_DIM, MAX_TOKENS + 1, feature_size, conv_feature, w_t, w_l, w_r, b_conv,
                        w_h, b_h, USE_GPU, embeddings)

    model.cuda()
    decay_ratio = 0.95
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-4, betas=[0.9, 0.999], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)
    loss_function = RankLoss(margin=1)

    allnodes, allchildren = sampling.gen_samples(trees, tbembeddings, lookup)

    # print(train_data)
    print('Start training...')

    # train_data_t, val_data_t = train_data, val_data
    train_loss_ = []
    val_loss_ = []

    cnt = 0
    cnt_v = 0

    PATH = 'tbcnn_codesearch_0126'
    # checkpoint = torch.load(PATH)
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(torch.load(PATH))
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        # training epoch
        total_loss = 0.0
        total = 0.0

        print("Epoch: %d" % epoch)

        # train_data_t = train_data_t.sample(frac=1)
        bs = BATCH_SIZE
        model.train()
        
        print("datasize:")
        print(len(allnodes))
        print(len(allchildren))
        print(len(train))
        print(len(test))
        for i in tqdm(range(0, len(train), bs)):
            if i + bs > len(train):
                bs = len(train) - i
            # batch = get_batch(test, i, bs)
            batch = load_javadata.get_batch(allnodes, allchildren, i, bs,  train_csv, train)

            node1, node2, ch1, ch2, doc1, doc2 = batch
            if USE_GPU:
                node1 = torch.tensor(node1).cuda()
                node2 = torch.tensor(node2).cuda()

                ch1 = torch.tensor(ch1).cuda()
                ch2 = torch.tensor(ch2).cuda()
                # doc1 = torch.tensor(doc1).cuda()
                # doc2 = torch.tensor(doc2).cuda()
            model.zero_grad()

            model.batch_size = len(doc1)
            code_vec, nl_vec = model(node1, ch1, doc1)
            neg_code_vec, neg_nl_vec = model(node2, ch2, doc2)

            loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)
            loss.backward()
            optimizer.step()

            #writer.add_scalar('Train/Loss', loss.item(), cnt)
            cnt = cnt + 1

            total_loss += float(loss.item())
            total += len(node2)
            if i % 1000 == 0:
                print("batch_loss: %.4f   avg_loss: %.4f" % (loss.item(), total_loss / total))

        train_loss_.append(total_loss / total)
        torch.save(model.state_dict(), modelpt)
        total_loss = 0.0
        total = 0.0


        code_vecs = []
        nl_vecs = []

        model.eval()
        v_bs = BATCH_SIZE
        for i in tqdm(range(0, len(test), v_bs)):
            if i + v_bs > len(test):
                v_bs = len(test) - i
            batch = load_javadata.get_testbatch(allnodes, allchildren, i, 1,  test_csv, test, len(train))

            node1, node2, ch1, ch2, doc1, doc2 = batch

            if USE_GPU:
                node1 = torch.tensor(node1).cuda()
                node2 = torch.tensor(node2).cuda()

                ch1 = torch.tensor(ch1).cuda()
                ch2 = torch.tensor(ch2).cuda()
                # doc1 = torch.tensor(doc1).cuda()
                # doc2 = torch.tensor(doc2).cuda()

            model.batch_size = v_bs
            with torch.no_grad():
                code_vec, nl_vec = model(node1, ch1, doc1)
                code_vecs.extend(code_vec.cpu().numpy())
                nl_vecs.extend(nl_vec)

                neg_code_vec, neg_nl_vec = model(node2, ch2, doc2)
                loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)

                total_loss += float(loss.item())
                total += len(node2)

                # writer.add_scalar('validation/Loss', loss.item(), cnt_v)
                cnt_v = cnt_v + 1

        val_loss_.append(total_loss / total)

        poolsize = len(test)
        accs, mrrs = [], []
        for i in range(poolsize):
            nl_vec_rep = nl_vecs[i].expand(poolsize, -1)
            n_results = 10
            sims = F.cosine_similarity(torch.from_numpy(np.array(code_vecs)).to(nl_vec_rep.device),
                                       nl_vec_rep).cpu().numpy()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))

        acc = np.mean(accs)
        mrr = np.mean(mrrs)
        # writer.add_scalar('validation/MRR', mrr, epoch)
        # writer.add_scalar('validation/ACC', acc, epoch)

        poolsize = 999
        accs_999, mrrs_999 = [], []
        for i in range(len(test)):
            nl_vec_rep = nl_vecs[i].expand(poolsize, -1)
            pdd = pd.DataFrame(np.array(code_vecs))
            new = pd.concat([pdd[0:i], pdd[i + 1:]], axis=0).sample(n=poolsize - 1)
            new = pd.concat([pdd[i:i + 1], new], axis=0)

            n_results = 999
            sims = F.cosine_similarity(torch.from_numpy(new.values).to(nl_vec_rep.device), nl_vec_rep).cpu().numpy()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            predict_acc = predict[:10]
            predict_acc = [int(k) for k in predict_acc]
            # predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [0]
            accs_999.append(ACC(real, predict_acc))
            mrrs_999.append(MRR(real, predict))

        acc_999 = np.mean(accs_999)
        mrr_999 = np.mean(mrrs_999)

        print(
            '[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f, Validation MRR: %.4f, Validation ACC: %.4f, '
            'Validation 999 MRR: %.4f, validation 999 ACC:%.4f'
            % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch], mrr, acc, mrr_999, acc_999))

        scheduler.step()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)
