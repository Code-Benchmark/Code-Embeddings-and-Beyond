import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .statistic import calculate, old_calculate
from trainer.statistic import calculate, old_calculate


class Trainer:
    # def __init__(self, args, model, train_data, valid_data, valid_infer_data, test_infer_data, t_vocab):
    def __init__(self, args, model, train_data, valid_data=None, test_data=None, valid_infer_data=None, test_infer_data=None):
        self.args = args
        cuda_condition = torch.cuda.is_available() and self.args.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        if cuda_condition and torch.cuda.device_count() > 1:
            self.wrap = True
            model = nn.DataParallel(model)
        else:
            self.wrap = False
        self.model = model.to(self.device)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.valid_infer_data = valid_infer_data
        self.test_infer_data = test_infer_data
        self.optim = torch.optim.Adamax(self.model.parameters()) #, lr=self.args.lr, weight_decay=self.args.weight_decay
        self.writer_path = '{}_{}_{}'.format('relation' if args.relation_path else 'Naive', args.dataset,
                                             datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        print(self.writer_path)
        os.mkdir(os.path.join('run', self.writer_path))
        self.tensorboard_writer = None
        self.writer = open(os.path.join('run', self.writer_path, 'experiment.txt'), 'w')
        print(self.args, file=self.writer, flush=True)
        self.iter = -1
        # self.t_vocab = t_vocab
        self.best_epoch, self.best_f1 = 0, float('-inf')
        self.accu_steps = self.args.accu_batch_size // self.args.batch_size
        #self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion = nn.BCELoss()
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
        self.iteration(epoch, self.train_data, 'train')

    def test(self, epoch, type):
        if type == 'valid':
            self.iteration(epoch, self.valid_data, type, train=False)
        elif type == 'test':
            self.iteration(epoch,self.test_data, type, train=False)

    def label_smoothing_loss(self, logits, targets, eps=0, reduction='mean'):
        if eps == 0:
            return self.criterion(logits, targets)
        K = logits.shape[-1]
        one_hot_target = F.one_hot(targets, num_classes=K)
        l_targets = (one_hot_target * (1 - eps) + eps / K).detach()
        loss = -(logits * l_targets).sum(-1).masked_fill(targets == 0, 0.0)
        if reduction == 'mean':
            return loss.sum() / torch.count_nonzero(targets)
        elif reduction == 'sum':
            return loss.sum()
        return loss

    def iteration(self, epoch, data_loader, type, train=True):
        # str_code = "train" if train else "valid"
        str_code = type
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        total_acc = 0
        total = 0
        predicts = []
        trues = []
        if train:
            self.optim.zero_grad()
        for i, data in data_iter:
            data = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in data.items()}
            if train:
                cur_labels = (data['label'] == 1).float()
                self.model.train()
                
                out = self.model(data)

                loss = self.criterion(out, cur_labels)
                # print(loss)
                loss.backward()

                if (i + 1) % self.accu_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad()
                predicted = (out.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                trues.extend(cur_labels.cpu().numpy())
                # total_acc += (predicted == data['label']).sum()
                # total += self.args.batch_size
            else:
                cur_labels = (data['label'] >= 1).float()
                self.model.eval()
                with torch.no_grad():
                    out = self.model(data)
                    loss = self.criterion(out, cur_labels)
                    predicted = (out.data > 0.5).cpu().numpy()
                    predicts.extend(predicted)
                    trues.extend(cur_labels.cpu().numpy())
                    # loss = self.criterion(out.view(out.shape[0] * out.shape[1], -1),
                    #                       data['f_target'].view(-1))  # avg at every step
            avg_loss += loss.item()

        avg_loss = avg_loss / len(data_iter)
        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        print("EP%d_%s, avg_loss=%f, Precision=%3f, Recall=%3f, F1=%3f" %
              (epoch, str_code, avg_loss, precision, recall, f1), file=self.writer, flush=True)
        print('-------------------------------------', file=self.writer, flush=True)

        if self.args.save and train:
            save_dir = './checkpoint'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.model.state_dict(),
                       os.path.join(save_dir, "{}_{}.pth".format(self.writer_path, epoch)))

