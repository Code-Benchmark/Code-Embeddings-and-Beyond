import pandas as pd
import torch
import time
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from model import SearchModel
from loss import RankLoss
import random
import tensorflow as tf
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2 = [], []
    fp_x1,fp_x2 = [], []

    for _, item in tmp.iterrows():
        x1.append(eval(item['code_ids_']))
        x2.append(eval(item['doc_ids_']))
        fp_x1.append(eval(item['code_ids_']))
        fp_x2.append(eval(item['fp_doc_ids_']))
        #labels.append([item['label']])
    return x1, x2, fp_x1, fp_x2    # , torch.FloatTensor(labels)


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


if __name__ == '__main__':

    root = 'data/'
    lang = 'java'

    print("Train for ", str.upper(lang))
    train_data = pd.read_csv('./data/train_data_0318.csv')
    val_data = pd.read_csv('./data/test_data_0318.csv')

    print(len(train_data))  # 258014
    print(len(val_data))   # 10027s


    word2vec = Word2Vec.load("./data/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    EPOCHS = 100
    VALID_BATCH_SIZE = 999
    BATCH_SIZE = 32
    USE_GPU = True
    code_model = "Transformer"
    writer = SummaryWriter('log/0318_Transformer')

    # model = SearchModel(code_model, EMBEDDING_DIM, MAX_TOKENS+1, USE_GPU, embeddings)
    model = SearchModel(code_model, EMBEDDING_DIM, MAX_TOKENS + 1, USE_GPU, embeddings)
    model.cuda()
    decay_ratio = 0.95
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-4, betas=[0.9, 0.999], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)
    loss_function = RankLoss(margin=1)

    print(train_data)
    print('Start training...')

    train_data_t, val_data_t = train_data, val_data
    train_loss_ = []
    val_loss_ = []

    cnt = 0
    cnt_v = 0

    PATH = './models/model_Transformer_0318.pkl'

    for epoch in range(EPOCHS):
        start_time = time.time()
        # training epoch
        total_loss = 0.0
        total = 0.0

        print("Epoch: %d" % epoch)

        # train_data_t = train_data_t.sample(frac=1)

        model.train()
        bs = BATCH_SIZE
        for i in tqdm(range(0, len(train_data_t), bs)):
            if i + bs > len(train_data_t):
                bs = len(train_data_t) - i
            batch = get_batch(train_data_t, i, bs)

            code_1, doc_1, code_2, doc_2 = batch

            model.zero_grad()

            model.batch_size = len(code_1)
            code_vec, nl_vec = model(code_1, doc_1)
            neg_code_vec, neg_nl_vec = model(code_2, doc_2)

            loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), cnt)
            cnt = cnt + 1

            total_loss += float(loss.item())
            total += len(code_1)

            print("batch_loss: %.4f   avg_loss: %.4f" % (loss.item(), total_loss/total))

        train_loss_.append(total_loss / total)

        scheduler.step()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)

        total_loss = 0.0
        total = 0.0

        code_vecs = []
        nl_vecs = []

        model.eval()
        v_bs = BATCH_SIZE
        for i in tqdm(range(0, len(val_data_t), v_bs)):
            if i + v_bs > len(val_data_t):
                v_bs = len(val_data_t) - i
            batch = get_batch(val_data_t, i, v_bs)
            code_1, doc_1, code_2, doc_2 = batch

            model.batch_size = v_bs
            with torch.no_grad():
                code_vec, nl_vec = model(code_1, doc_1)
                code_vecs.extend(code_vec.cpu().numpy())
                nl_vecs.extend(nl_vec)

                neg_code_vec, neg_nl_vec = model(code_2, doc_2)
                loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)

                total_loss += float(loss.item())
                total += len(code_1)

                writer.add_scalar('validation/Loss', loss.item(), cnt_v)
                cnt_v = cnt_v + 1

        val_loss_.append(total_loss / total)

        poolsize = len(val_data_t)
        accs, mrrs = [], []
        for i in range(poolsize):
            nl_vec_rep = nl_vecs[i].expand(poolsize, -1)
            n_results = 10
            sims = F.cosine_similarity(torch.from_numpy(np.array(code_vecs)).to(nl_vec_rep.device), nl_vec_rep).cpu().numpy()
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
        for i in range(len(val_data_t)):
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
            % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch], mrr, acc, mrr_999, acc_999))




