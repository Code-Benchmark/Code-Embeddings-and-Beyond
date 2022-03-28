import pandas as pd
import torch
from tqdm import tqdm
import time
from models import SearchModel
from models import RankLoss
import numpy as np
from tensorboardX import SummaryWriter
from gensim.models.word2vec import Word2Vec
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import json

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

    data = pd.read_csv('./data/new_data_0316.csv')
    train_data = data[data['partition'] =='train']
    val_data = data[data['partition'] == 'test']
    # test_data = data[data['partition'] == 'test']

    data_json = json.loads(open('./data/ggnn_wiz_w2v.json', 'r').read())

    word2vec_doc = Word2Vec.load("./data/node_wiz_token_w2v_128").wv
    MAX_TOKENS = word2vec_doc.syn0.shape[0]
    EMBEDDING_DIM = word2vec_doc.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec_doc.syn0.shape[0]] = word2vec_doc.syn0
    # print(MAX_TOKENS_doc)

    EPOCHS = 100
    BATCH_SIZE = 64
    USE_GPU = True
    vocablen = MAX_TOKENS + 1   # java: 77410
    edge_type = 7
    # EMBEDDING_DIM = 200
    num_layers = 4
    device = torch.device('cuda:0')
    # MAX_TOKENS = 12097  # 固定的
    #def __init__(self, vocablen, embedding_dim, num_layers, device):
    model = SearchModel(vocablen, EMBEDDING_DIM, num_layers, device, embeddings)
    # model = lstm_model(MAX_TOKENS + 1, EMBEDDING_DIM)
    cnt = 0
    cnt_v = 0
    PATH = './model/model_search_0316.pkl'
    writer = SummaryWriter('log/0316')
    if USE_GPU:
        model.cuda()

    decay_ratio = 0.95
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=1e-4, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)
    # loss_function = torch.nn.CrossEntropyLoss()   # NLLLoss
    loss_function = RankLoss(margin=1)
    print('Done creating GGNN Model')

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
        model.train()
        for i in tqdm(range(0, len(train_data))):  #len(train_data)
            # if i+bs > len(train_data):
            #     bs = len(train_data) - i
            # batch = get_batch(train_data, i, bs)
            # i += BATCH_SIZE
            # train_inputs, train_labels = batch
            # train_label = train_data[i:i+1]['label'].values[0]

            # train_label_c = torch.FloatTensor([train_label]).cuda()
            train_code = data_json[str(train_data[i:i+1]['index'].values[0])]
            train_doc = eval(train_data[i:i+1]['doc_ids'].values[0])
            train_doc_fp = eval(train_data[i:i+1]['fp_doc_ids'].values[0])
            # if USE_GPU:
            #     train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            # model.batch_size = len(train_labels)
            model.batch_size = 1
            #model.hidden = model.init_hidden()
            code_vec, nl_vec = model(train_code, train_doc)
            neg_code_vec, neg_nl_vec = model(train_code, train_doc_fp)

            loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)
            # loss = loss_function(output, torch.arange(BATCH_SIZE, device=output.device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            writer.add_scalar('Train/Loss', loss.item(), cnt)
            cnt = cnt + 1
            # calc training acc
            total_loss += float(loss.item())
            total += 1
            print("item_loss: %.4f   avg_loss: %.4f" % (loss.item(), total_loss / total))
            # print("loss: %f   acc: %f" % (loss.item(), (predicted == train_labels).sum().float()/len(train_labels)))

        train_loss_.append(total_loss / total)

        scheduler.step()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        code_vecs = []
        nl_vecs = []

        bs = 1
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(val_data), bs)):
                # if i + bs > len(val_data):
                #     bs = len(val_data) - i

                val_code = data_json[str(val_data[i:i + 1]['index'].values[0])]
                val_doc = eval(val_data[i:i + 1]['doc_ids'].values[0])
                val_doc_fp = eval(val_data[i:i + 1]['fp_doc_ids'].values[0])
                # if USE_GPU:
                #     val_starts, val_paths, val_ends, val_masks, val_labels = \
                #         val_starts, val_paths, val_ends, val_masks, val_labels.cuda()

                code_vec, nl_vec = model(val_code, val_doc)
                code_vecs.extend(code_vec.cpu().numpy())
                nl_vecs.extend(nl_vec)

                neg_code_vec, neg_nl_vec = model(val_code, val_doc_fp)
                loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)

                total_loss += float(loss.item())
                total += 1

                writer.add_scalar('validation/Loss', loss.item(), cnt_v)
                cnt_v = cnt_v + 1
            # print("loss: %f   acc: %f" % (loss.item(), (predicted == val_labels).sum().float() / len(val_labels)))

            val_loss_.append(total_loss / total)

        poolsize = len(val_data)
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
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict_acc))
            mrrs.append(MRR(real, predict))

        acc = np.mean(accs)
        mrr = np.mean(mrrs)
        writer.add_scalar('validation/MRR', mrr, epoch)
        writer.add_scalar('validation/ACC', acc, epoch)

        poolsize = 999
        accs_999, mrrs_999 = [], []
        for i in range(len(val_data)):
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


