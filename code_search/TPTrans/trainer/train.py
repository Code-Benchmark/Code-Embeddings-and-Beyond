import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from .statistic import calculate, old_calculate

class RankLoss(nn.Module):
    def __init__(self, margin):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, nl_vec_pos, nl_vec_neg, code_vec_pos, code_vec_neg):
        return (self.margin - F.cosine_similarity(nl_vec_pos,
                                                  code_vec_pos) + F.cosine_similarity(
            nl_vec_neg, code_vec_pos)).clamp(min=1e-6).mean()


class Trainer:
    # def __init__(self, args, model, train_data, valid_data, valid_infer_data, test_infer_data, t_vocab):
    def __init__(self, args, model, train_data=None, test_data=None, t_vocab=None):
        self.args = args
        cuda_condition = torch.cuda.is_available() and self.args.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.wrap = False
        self.model = model.to(self.device)
        self.train_data = train_data
        self.test_data = test_data
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda epoch: 0.95 ** epoch)
        self.writer_path = '{}_{}_{}'.format('relation' if args.relation_path else 'Naive', args.dataset,
                                             datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        print(self.writer_path)
        os.mkdir(os.path.join('run', self.writer_path))
        # self.tensorboard_writer = SummaryWriter(os.path.join('run', self.writer_path))
        self.writer = open(os.path.join('run', self.writer_path, 'experiment.txt'), 'w')
        print(self.args, file=self.writer, flush=True)
        self.iter = -1
        self.t_vocab = t_vocab
        self.best_epoch, self.best_f1 = 0, float('-inf')
        self.accu_steps = self.args.accu_batch_size // self.args.batch_size
        # self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion = RankLoss(margin=1)
        self.unk_shift = self.args.unk_shift
        if self.args.relation_path or self.args.absolute_path:
            print(
                "Total Parameters: {}*1e6".format(sum([p.nelement() for _, p in self.model.named_parameters()]) // 1e6),
                file=self.writer, flush=True)
        else:
            model_parameters = []
            for name, param in self.model.named_parameters():
                if 'path' in name:
                    continue
                else:
                    model_parameters.append(param)
            print("Total Parameters: {}*1e6".format(sum([p.nelement() for p in model_parameters]) // 1e6),
                  file=self.writer, flush=True)

    def load(self, path):
        dic = torch.load(path, map_location='cpu')
        load_pre = ''
        model_pre = ''
        print(dic.keys())
        for key, _ in dic.items():
            if 'module.' in key:
                load_pre = 'module.'
            else:
                load_pre = ''
            break
        for key, _ in self.model.state_dict().items():
            if 'module.' in key:
                model_pre = 'module.'
            else:
                model_pre = ''
            break
        if load_pre == '' and model_pre == 'module.':
            temp_dict = dict()
            for key, value in dic.items():
                temp_dict[model_pre + key] = value
            dic = temp_dict
        elif model_pre == '' and load_pre == 'module.':
            temp_dict = dict()
            for key, value in dic.items():
                temp_dict[key.replace(load_pre, model_pre)] = value
            dic = temp_dict
        temp_dict = dict()
        ori_dic = self.model.state_dict()
        for key, value in dic.items():
            if key in ori_dic and ori_dic[key].shape == value.shape:
                temp_dict[key] = value
        dic = temp_dict
        print(dic.keys())
        for key, value in self.model.state_dict().items():
            if key not in dic:
                dic[key] = value
        self.model.load_state_dict(dic)
        print('Load Pretrain model => {}'.format(path))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)


    def ACC(self, real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + 1
        return sum / float(len(real))


    def MRR(self, real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + 1.0 / float(index + 1)
        return sum / float(len(real))

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        code_vecs = []
        nl_vecs = []
        if train:
            self.optim.zero_grad()
        for i, data in data_iter:
            data = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in data.items()}
            if train:
                self.model.train()
                code_vec, nl_vec, neg_nl_vec = self.model(data)

                loss = self.criterion(nl_vec, neg_nl_vec, code_vec, code_vec)
                accu_loss = loss / self.accu_steps
                accu_loss.backward()
                if (i + 1) % self.accu_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad()
                # self.optim.step()
                # self.optim.zero_grad()
                
            else:
                self.model.eval()
                with torch.no_grad():
                    code_vec, nl_vec, neg_nl_vec = self.model(data)
                    # code_vec, nl_vec = self.model(data)
                    # code_vecs.extend(code_vec.cpu().numpy())
                    # nl_vecs.extend(nl_vec.cpu().numpy())
                    # loss = self.criterion(nl_vec, neg_nl_vec, code_vec, code_vec)

        np.save('code_vec.npy',np.array(code_vecs))
        np.save('nl_vec.npy',np.array(nl_vecs))


        avg_loss += loss.item()
        if train:
            avg_loss = avg_loss / len(data_iter)
            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss, file=self.writer, flush=True)
            print('-------------------------------------', file=self.writer, flush=True)
            self.scheduler.step()
        else:
            poolsize = 999
            accs_999, mrrs_999 = [], []
            for i in range(len(nl_vecs)):
                nl_vec_rep = nl_vecs[i].expand(poolsize, -1)
                pdd = pd.DataFrame(np.array(code_vecs))
                new = pd.concat([pdd[0:i], pdd[i + 1:]], axis=0).sample(n=poolsize - 1)
                new = pd.concat([pdd[i:i + 1], new], axis=0)
        
                n_results = poolsize
                sims = F.cosine_similarity(torch.from_numpy(new.values).to(nl_vec_rep.device), nl_vec_rep).cpu().numpy()
                negsims = np.negative(sims)
                predict = np.argsort(negsims)
                predict_acc = predict[:10]
                predict_acc = [int(k) for k in predict_acc]
                predict = predict[:n_results]
                predict = [int(k) for k in predict]
                real = [0]
                if real[0] in predict_acc:
                accs_999.append(self.ACC(real, predict_acc))
                mrrs_999.append(self.MRR(real, predict))
            acc_999 = np.mean(accs_999)
            mrr_999 = np.mean(mrrs_999)
            avg_loss = avg_loss / len(data_iter)
            print("EP%d_%s, avg_loss=%.4f, 999MRR=%.4f, 999ACC=%.4f" % (epoch, str_code, avg_loss, mrr_999, acc_999), file=self.writer, flush=True)
            print('-------------------------------------', file=self.writer, flush=True)



        if self.args.save and train:
            save_dir = './checkpoint'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.model.state_dict(),
                       os.path.join(save_dir, "{}_{}.pth".format(self.writer_path, epoch)))

