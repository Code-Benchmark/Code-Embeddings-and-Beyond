from model import CodeSearchModel
from process_data import C_code
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
import random
import pandas as pd
import argparse
from model import RankLoss
import csv
import os
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


def eval_doc(x, i, bs):
    new_x = []
    for j in range(i, i+bs):
        new_x.append(eval(x[j]))
    return new_x


if __name__ == '__main__':
    BATCH_SIZE = 32
    # Load Data
    data = C_code()

    # "doc_ids","fp_doc_ids"
    train = pd.read_csv('./data/train_data_0304.csv',header=None)
    train.columns = ["id","context","doc_ids","fp_doc_ids"]
    x_code = data.load_data('train')
    x_doc = train['doc_ids'].values[1:]
    fp_x_doc = train['fp_doc_ids'].values[1:]
    print('train dataset code: ' + str(np.array(x_code).shape))
    print('train dataset doc: ' + str(x_doc.shape))

    dev = pd.read_csv('./data/test_data_0304.csv',header=None)
    dev.columns = ["id","context","doc_ids","fp_doc_ids"]
    x_code_val = data.load_data('test')
    x_doc_val = dev['doc_ids'].values[1:]
    fp_x_doc_val = dev['fp_doc_ids'].values[1:]
    print('valid dataset: ' + str(np.array(x_code_val).shape))
    print('valid dataset doc: ' + str(x_doc_val.shape))


    word2vec = Word2Vec.load("./data/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    print('Done Data Loader')


    # Load Model
    nodes_dim = 999296  #999178
    paths_dim = 304280
    embedding_dim = EMBEDDING_DIM
    code_vector_size = 3 * embedding_dim
    # output_dim = 2
    dropout = 0.25
    USE_GPU = True
    max_token = MAX_TOKENS
    # max_path_node = 208
    max_nl_len = 20
    # nodes_dim, paths_dim, embedding_dim, code_vector_size, vocab_size_doc, vocab_size_path=None, pretrained_weight
    model = CodeSearchModel(nodes_dim, paths_dim, embedding_dim, code_vector_size, max_token+1, embeddings)
    # model = PSCSNetwork(max_token + 1, max_path_node, embedding_dim)

    if USE_GPU:
        model.cuda()

    # optimizer = torch.optim.Adamax(model.parameters())
    # loss_function = torch.nn.NLLLoss()
    decay_ratio = 0.95
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=1e-4, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)
    # loss_function = torch.nn.CrossEntropyLoss()   # NLLLoss
    loss_function = RankLoss(margin=1)
    print('Done creating Code2VecModel')

    EPOCHS = 100
    writer = SummaryWriter('log/0316')
    PATH = './model/model_wizout_w2v_0316.pkl'
    # checkpoint = torch.load(PATH)
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['model_state_dict'])

    cnt = 0
    cnt_v = 0

    print('Start training...')
    bs = BATCH_SIZE
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    for epoch in range(EPOCHS):
        # train
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        bs = BATCH_SIZE
        model.train()
        for i in tqdm(range(0, len(x_doc), BATCH_SIZE)):
            if i+bs > len(x_doc):
                bs = len(x_doc) - i
            x_code_true = []
            for sample in x_code[i:i+bs]:
                # if len(sample) > 200:
                #     sample = random.sample(sample, 200)
                # else:
                #     sample = sample[0:200]
                x_code_true.append(sample)
            x_code_true = np.array(x_code_true).swapaxes(1, 2)

            train_starts = torch.LongTensor(x_code_true[:,0,:].tolist()).cuda()
            # train_starts = x_code[0][i:i + bs]
            train_paths = torch.LongTensor(x_code_true[:,1,:].tolist()).cuda()
            # train_ends = x_code[2][i:i + bs]
            train_ends = torch.LongTensor(x_code_true[:,2,:].tolist()).cuda()
            train_masks = torch.FloatTensor(x_code_true[:,3,:].tolist()).cuda()
            x_doc_true = [eval(doc) for doc in x_doc[i:i + bs]]
            fp_x_doc_true = [eval(doc) for doc in fp_x_doc[i:i + bs]]
            # train_doc = eval_doc(x_doc, i, bs)
            # train_labels = torch.LongTensor(y[i: i + bs]).cuda()

            # if USE_GPU:
            #     train_starts, train_paths, train_ends, train_masks, train_labels = \
            #         train_starts, train_paths, train_ends, train_masks, train_labels.cuda()
            optimizer.zero_grad()

            code_vec, nl_vec = model(train_starts, train_paths, train_ends, train_masks, x_doc_true)
            neg_code_vec, neg_nl_vec = model(train_starts, train_paths, train_ends, train_masks, fp_x_doc_true)

            loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)
            # loss = loss_function(output, torch.arange(BATCH_SIZE, device=output.device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), cnt)
            cnt = cnt + 1
            # calc training acc
            total_loss += float(loss.item())
            total += len(x_doc_true)
            print("batch_loss: %.4f   avg_loss: %.4f" % (loss.item(), total_loss/total))
            # print("loss: %f   acc: %f" % (loss.item(), (predicted == train_labels).sum().float()/len(train_labels)))

        train_loss_.append(total_loss / total)
        scheduler.step()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)

        # validation
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        code_vecs = []
        nl_vecs = []
        model.eval()
        bs = BATCH_SIZE
        for i in tqdm(range(0, len(x_doc_val), bs)):
            if i + bs > len(x_doc_val):
                bs = len(x_doc_val) - i
            x_code_val_true = []
            for sample in x_code_val[i:i + bs]:
                if len(sample) > 200:
                    sample = random.sample(sample, 200)
                else:
                    sample = sample[0:200]
                x_code_val_true.append(sample)
            x_code_val_true = np.array(x_code_val_true).swapaxes(1, 2)
            val_starts = torch.LongTensor(x_code_val_true[:, 0, :].tolist()).cuda()
            # train_starts = x_code[0][i:i + bs]
            val_paths = torch.LongTensor(x_code_val_true[:, 1, :].tolist()).cuda()
            # train_ends = x_code[2][i:i + bs]
            val_ends = torch.LongTensor(x_code_val_true[:, 2, :].tolist()).cuda()
            val_masks = torch.FloatTensor(x_code_val_true[:, 3, :].tolist()).cuda()
            val_x_doc_true = [eval(doc) for doc in x_doc_val[i:i+bs]]
            fp_val_x_doc_true = [eval(doc) for doc in fp_x_doc_val[i:i + bs]]

            # if USE_GPU:
            #     val_starts, val_paths, val_ends, val_masks, val_labels = \
            #         val_starts, val_paths, val_ends, val_masks, val_labels.cuda()

            with torch.no_grad():
                code_vec, nl_vec = model(val_starts, val_paths, val_ends, val_masks, val_x_doc_true)
                code_vecs.extend(code_vec.cpu().numpy())
                nl_vecs.extend(nl_vec)

                neg_code_vec, neg_nl_vec = model(val_starts, val_paths, val_ends, val_masks, fp_val_x_doc_true)
                loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)

                total_loss += float(loss.item())
                total += len(x_doc_val)

                writer.add_scalar('validation/Loss', loss.item(), cnt_v)
                cnt_v = cnt_v + 1
            #print("loss: %f   acc: %f" % (loss.item(), (predicted == val_labels).sum().float() / len(val_labels)))

        val_loss_.append(total_loss / total)

        poolsize = len(x_code_val)
        accs, mrrs = [], []
        for i in range(poolsize):
            nl_vec_rep = nl_vecs[i].expand(poolsize, -1)
            n_results = 10
            sims = F.cosine_similarity(torch.from_numpy(np.array(code_vecs)).to(nl_vec_rep.device),
                                       nl_vec_rep).cpu().numpy()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            predict_acc = predict[:n_results]
            predict_acc = [int(k) for k in predict_acc]
            real = [i]
            predict_mrr = predict[:poolsize]
            predict_mrr = [int(k) for k in predict_mrr]
            accs.append(ACC(real, predict_acc))
            mrrs.append(MRR(real, predict_mrr))

        acc = np.mean(accs)
        mrr = np.mean(mrrs)
        writer.add_scalar('validation/MRR', mrr, epoch)
        writer.add_scalar('validation/ACC', acc, epoch)

        poolsize = 999
        accs_999, mrrs_999 = [], []
        for i in range(len(x_code_val)):
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
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [0]
            accs_999.append(ACC(real, predict_acc))
            mrrs_999.append(MRR(real, predict))

        acc_999 = np.mean(accs_999)
        mrr_999 = np.mean(mrrs_999)

        print(
            '[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f, Validation MRR: %.4f, Validation ACC: %.4f, '
            'Validation 999 MRR: %.4f, validation 999 ACC:%.4f'
            % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch], mrr, acc,mrr_999,acc_999))

