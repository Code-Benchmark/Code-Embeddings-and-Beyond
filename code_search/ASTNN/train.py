import pandas as pd
import torch
import time
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
#from astnn.search.model import BatchProgramCC
from model import ASTNN4Search
from model import RankLoss
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
    fp_x1, fp_x2 = [], []

    for _, item in tmp.iterrows():
        x1.append(item['code_ids'])
        x2.append(eval(item['doc_ids']))
        fp_x1.append(item['code_ids'])
        fp_x2.append(eval(item['fp_doc_ids']))
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
    train_data = pd.read_pickle(root+lang+'/train_data_0304.pkl')
    val_data = pd.read_pickle(root+lang+'/test_data_0304.pkl')

    print(len(train_data))
    print(len(val_data))
    #print(len(test_data))

    word2vec = Word2Vec.load(root+lang+"/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    EPOCHS = 100
    BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    # MAX_TOKENS_doc = 18912
    USE_GPU = True
    writer = SummaryWriter('log/0304')

    # f = open("result.txt", "a")
    EMBEDDING_DIM = 128
    EMBEDDING_DIM_doc = 128
    # MAX_TOKENS_code = 115061  #49429 #18898    #49429
    # MAX_TOKENS_doc = 115061  #49429 #18898  #49429
    model = ASTNN4Search(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS+1,  # +1 because nan = max_token
                            ENCODE_DIM, BATCH_SIZE, USE_GPU, embeddings)
    model.cuda()
    model.hidden = model.init_hidden()
    decay_ratio = 0.95
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=1e-4, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)
    #loss_function = torch.nn.CrossEntropyLoss()   # NLLLoss
    loss_function = RankLoss(margin=1)

    print(train_data)
    #precision, recall, f1 = 0, 0, 0
    print('Start training...')

    #train_data_t, val_data_t, test_data_t = train_data, val_data, test_data
    train_data_t, val_data_t = train_data, val_data
    train_loss_ = []
    train_acc_ = []
    val_loss_ = []
    val_acc_ = []

    cnt = 0
    cnt_v = 0

    PATH = './model/model_0304.pkl'
    # checkpoint = torch.load(PATH)
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(EPOCHS):

        start_time = time.time()

        # training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0

        #avg_loss = 0.0
        # predicts = []
        # trues = []
        print("Epoch: %d" % epoch)

        train_data_t = train_data_t.sample(frac=1)

        model.train()

        bs = BATCH_SIZE
        for i in tqdm(range(0, len(train_data_t), bs)):
            if i + bs > len(train_data_t):
                bs = len(train_data_t) - i
            batch = get_batch(train_data_t, i, BATCH_SIZE)
            #i += BATCH_SIZE
            #train1_inputs, train2_inputs, train_labels = batch
            code_1, doc_1, code_2, doc_2 = batch
            #if USE_GPU:
                # train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()
                # train1_inputs, train2_inputs = train1_inputs, train2_inputs.cuda()
            #    train1_inputs = train1_inputs.cuda()

            model.zero_grad()
            model.batch_size = bs
            model.hidden = model.init_hidden()

            code_vec, nl_vec = model(code_1, doc_1)
            neg_code_vec, neg_nl_vec = model(code_2, doc_2)

            #loss = loss_function(output, train_labels.squeeze(1).to(dtype=torch.long))
            loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)
            #loss = loss_function(output, torch.arange(BATCH_SIZE, device=output.device))
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), cnt)
            cnt = cnt + 1

            total_loss += float(loss.item())
            total += len(code_1)

            print("batch_loss: %.4f   avg_loss: %.4f" % (loss.item(), total_loss/total))

        #print(total)

        train_loss_.append(total_loss / total)
        scheduler.step()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)
       # train_acc_.append(accuracy_score(predicts, trues))

        total_loss = 0.0
        total = 0.0
        # predicts = []
        # trues = []
        code_vecs = []
        nl_vecs = []
        VALID_BATCH_SIZE = 32

        model.eval()
        v_bs = BATCH_SIZE
        for i in tqdm(range(0, len(val_data_t), v_bs)):
            if i + v_bs > len(val_data_t):
                v_bs = len(val_data_t) - i
            batch = get_batch(val_data_t, i, v_bs)
            code_1, doc_1, code_2, doc_2 = batch
            #if USE_GPU:
                #val1_inputs, val2_inputs = val1_inputs, val2_inputs.cuda()
                #val1_inputs = val1_inputs.cuda()

            model.batch_size = v_bs
            model.hidden = model.init_hidden()
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
        for i in range(len(val_data_t)):
            nl_vec_rep = nl_vecs[i].expand(poolsize, -1)
            pdd = pd.DataFrame(np.array(code_vecs))
            new = pd.concat([pdd[0:i], pdd[i + 1:]], axis=0).sample(n=poolsize - 1)
            new = pd.concat([pdd[i:i + 1], new], axis=0)

            n_results = 999
            sims = F.cosine_similarity(torch.from_numpy(['new.values']).to(nl_vec_rep.device), nl_vec_rep).cpu().numpy()
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

