import pandas as pd
import torch
import time
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from Search.TreeCaps.model.model import SearchModel
from Search.TreeCaps.model.loss import RankLoss

import pickle
import Search.TreeCaps.sampling as sampling
import Search.TreeCaps.load_javadata as load_javadata

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


def load_data(rootfile, treefile):
    trees_file = rootfile + treefile

    with open(trees_file, 'rb') as fh:
        trees = pickle.load(fh)

    # id = list(set(id))
    return trees


def train(args, train_data, test_data):

    root = args.dataset_directory
    lang = 'java'

    print("Train for ", str.upper(lang))


    word2vec = Word2Vec.load(root+"/node_wiz_token_w2v_128").wv
    MAX_TOKENS = len(word2vec.vectors)
    EMBEDDING_DIM = word2vec.vector_size
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:MAX_TOKENS] = word2vec.vectors

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    USE_GPU = args.USE_GPU
    conv_size = 64
    num_conv = 8
    num_outputs = 1
    num_dims = 16
    num_feats = 128
    conv_feature = 64
    cuda1 = torch.device('cuda:0')
    num_feats = 128

    p1 = torch.tensor(num_feats).cuda()
    p2 = torch.tensor(1).cuda()
    p3 = torch.tensor(conv_feature).cuda()
    train_csv, test_csv, all_csv = load_javadata.read_train_data(root,root + '/example.pkl', root + '/example.csv')

    trees = load_data(root, '/example_trees.pkl')
    for i in range(1, len(trees)):
        if trees[i]['id'] == 0:
            print(i)
            break
    train = trees[:len(train_csv)]
    test = trees[len(train_csv):]

    Wemd = torch.nn.init.uniform(torch.zeros((MAX_TOKENS, num_feats), dtype=torch.float32, device=cuda1))
    model = SearchModel(EMBEDDING_DIM, MAX_TOKENS, p1, p2, p3,conv_size, num_conv, num_feats, num_dims, num_outputs, Wemd, MAX_TOKENS,pretrained_weight=embeddings)
    model.cuda()
    decay_ratio = 0.95
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-4, betas=[0.9, 0.999], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)
    loss_function = RankLoss(margin=1)
    allnodes, allchildren = sampling.gen_samples(root,trees)

    print('Start training...')
    train_loss_ = []
    val_loss_ = []
    cnt = 0
    cnt_v = 0

    PATH = args.model_path
    
    for epoch in range(EPOCHS):
        print("Epoch: %d" % epoch)
        start_time = time.time()
        total_loss = 0.0
        total = 0.0

        bs = BATCH_SIZE
        model.train()
        for i in tqdm(range(0, len(train_csv), bs)):
            if i + bs > len(train):
                bs = len(train) - i

            batch = load_javadata.get_batch(allnodes, allchildren, i, bs,  train_csv, train)
            node1, node2, ch1, ch2, doc1, doc2 = batch

            if USE_GPU:
                node1 = torch.tensor(node1).cuda()
                node2 = torch.tensor(node2).cuda()
                ch1 = torch.tensor(ch1).cuda()
                ch2 = torch.tensor(ch2).cuda()
            model.zero_grad()
            model.batch_size = len(doc1)
            code_vec, nl_vec = model(node1, ch1, doc1)
            neg_code_vec, neg_nl_vec = model(node2, ch2, doc2)
            loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)
            loss.backward()
            optimizer.step()
            cnt = cnt + 1
            total_loss += float(loss.item())
            total += len(node2)
            if i % 1000 == 0:
                print("batch_loss: %.4f   avg_loss: %.4f" % (loss.item(), total_loss / total))

        train_loss_.append(total_loss / total)
        total_loss = 0.0
        total = 0.0
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()
                    }, PATH)

        code_vecs = []
        nl_vecs = []

        model.eval()
        v_bs = BATCH_SIZE
        for i in tqdm(range(0, len(test_csv), v_bs)):
            if i + v_bs > len(test):
                v_bs = len(test) - i
            batch = load_javadata.get_testbatch(allnodes, allchildren, i, 1,  test_csv, test, len(train_csv))

            node1, node2, ch1, ch2, doc1, doc2 = batch

            if USE_GPU:
                node1 = torch.tensor(node1).cuda()
                node2 = torch.tensor(node2).cuda()
                ch1 = torch.tensor(ch1).cuda()
                ch2 = torch.tensor(ch2).cuda()

            model.batch_size = v_bs
            with torch.no_grad():
                code_vec, nl_vec = model(node1, ch1, doc1)
                code_vecs.extend(code_vec.cpu().numpy())
                nl_vecs.extend(nl_vec)

                neg_code_vec, neg_nl_vec = model(node2, ch2, doc2)
                loss = loss_function(nl_vec, neg_nl_vec, code_vec, neg_code_vec)

                total_loss += float(loss.item())
                total += len(node2)

                cnt_v = cnt_v + 1

        val_loss_.append(total_loss / total)

        poolsize = len(test)
        accs, mrrs = [], []
        for i in range(poolsize):
            nl_vec_rep = nl_vecs[i].expand(poolsize, -1)
            n_results = 10
            sims = F.cosine_similarity(torch.from_numpy(np.array(code_vecs)).to(nl_vec_rep.device),nl_vec_rep).cpu().numpy()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))

        acc = np.mean(accs)
        mrr = np.mean(mrrs)
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



