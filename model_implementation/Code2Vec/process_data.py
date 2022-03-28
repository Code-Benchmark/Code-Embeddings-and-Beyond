import pandas as pd
import os
import sys
import numpy as np
import random


class C_code():
    def __init__(self):
        self.base_path = './data/'
        self.rawData_path = self.base_path + 'rawData'
        self.outData_path = self.base_path + 'outData_test'
        self.tempData_path = self.base_path + 'tempData'

    # execute for the first time
    def preprocess_data(self):
        if not os.path.exists(self.outData_path):
            os.mkdir(self.outData_path)
        if not os.path.exists(self.tempData_path):
            os.mkdir(self.tempData_path)

        def processLine(row):
            id = row['id']
            code = row['code']
            with open(os.path.join(self.tempData_path, str(id) + '.c'), 'w', encoding='utf-8') as f:
                f.write(code)

        # for root, dirs, files in os.walk(self.rawData_path):
        #    for f in files:
        cur = pd.read_pickle(os.path.join(self.base_path, 'programs.pkl'))
        cur.columns = ['id', 'code', 'label']
        cur.apply(processLine, axis=1)

        ret = os.system(
            r'java -jar cli.jar pathContexts --lang c --project ' + self.tempData_path + ' --output ' + self.outData_path +
            r' --maxH 8 --maxW 2 --maxContexts ' + str(200) + ' --maxTokens ' + str(1301136) +
            ' --maxPaths ' + str(911417))

        assert ret == 0
        print("Extract Code Paths done!")

    # split data for training, developing and testing
    def split_data(self):
        data = open(os.path.join(self.outData_path, 'c/path_contexts.csv'), 'r', encoding='UTF-8').readlines()
        # data = pd.read_csv(os.path.join(self.outData_path, 'c/path_contexts.csv'))
        data_num = len(data)
        data = pd.DataFrame(np.array(data))
        ratio = '3:1:1'
        ratios = [int(r) for r in ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        print(train_split)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)
        print(val_split)
        data = data.sample(frac=1, random_state=666)
        train = data[:train_split]
        dev = data[train_split:val_split]
        test = data[val_split:]

        train_file_path = self.base_path + 'train.csv'
        train.to_csv(train_file_path, index=False, header=None, sep=",")

        dev_file_path = self.base_path + 'valid.csv'
        dev.to_csv(dev_file_path, index=False, header=None, sep=",")

        test_file_path = self.base_path + 'test.csv'
        test.to_csv(test_file_path, index=False, header=None, sep=",")

    def load_data(self, tag):
        source_tokens = []
        path_tokens = []
        target_tokens = []
        context_valid_masks = []

        with open(os.path.join(self.base_path, tag + '_code.csv'), 'r', encoding='UTF-8') as f:
            p = f.readlines()
            for line in p:
                line = line.replace("\n", "").replace("\"", "").split(' ')

                cur_source_tokens = []
                cur_path_tokens = []
                cur_target_tokens = []
                cur_context_masks = []

                cnt = 0
                for path in line[1:]:
                    source_token, path_token, target_token = map(int, path.strip().split(','))
                    # print(source_token, path_token, target_token)

                    cur_source_tokens.append(source_token)
                    cur_path_tokens.append(path_token)
                    cur_target_tokens.append(target_token)
                    cur_context_masks.append(1)
                    cnt = cnt + 1

                cur_source_tokens = np.pad(cur_source_tokens, (0, 200 - cnt), 'constant',
                                           constant_values=(0, 0))
                cur_path_tokens = np.pad(cur_path_tokens, (0, 200 - cnt), 'constant',
                                         constant_values=(0, 0))
                cur_target_tokens = np.pad(cur_target_tokens, (0, 200 - cnt), 'constant',
                                           constant_values=(0, 0))
                cur_context_masks = np.pad(cur_context_masks, (0, 200 - cnt), 'constant',
                                           constant_values=(0, 0))

                source_tokens.append(cur_source_tokens)
                path_tokens.append(cur_path_tokens)
                target_tokens.append(cur_target_tokens)
                context_valid_masks.append(cur_context_masks)
        return source_tokens, path_tokens, target_tokens, context_valid_masks

    def get_label(self, tag):
        with open(os.path.join(self.base_path, tag + '_label.csv'), 'r') as f:
            p = f.readlines()
            p = [int(i)-1 for i in p]
        return p


if __name__ == "__main__":
    os.chdir(sys.path[0])
    dataTool = C_code()
    dataTool.preprocess_data()
