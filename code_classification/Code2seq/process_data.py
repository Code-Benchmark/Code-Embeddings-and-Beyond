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
        cur = pd.read_pickle('../data.pkl')
        cur.columns = ['id', 'code', 'label']
        cur.apply(processLine, axis=1)

        ret = os.system(
            r'java -jar cli.jar pathContexts --lang c --project ' + self.tempData_path + ' --output ' + self.outData_path +
            r' --maxH 8 --maxW 2 --maxContexts ' + str(200) + ' --maxTokens ' + str(1301136) +
            ' --maxPaths ' + str(911417))

        assert ret == 0
        print("Extract Code Paths done!")

    def load_data_2(self, tag):
        path_dict = np.load('./data/path_dict.npy', allow_pickle=True).item()
        node_dict = np.load('./data/node_dict.npy', allow_pickle=True).item()
        context = []
        with open(os.path.join('./data/', 'sample_' + tag + '_code.csv'), 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.split(' ')
                cur_context = []
                cnt = 0
                for path in line[1:]:
                    source_token, path_token, target_token = map(int, path.strip().replace("\"", "").split(','))
                    source_token = [int(i) for i in node_dict[source_token].split(",")]
                    path_token = [int(i) for i in path_dict[path_token].split(",")]
                    target_token = [int(i) for i in node_dict[target_token].split(",")]
                    cur_context.append([source_token, path_token, target_token, 1])

                    cnt = cnt + 1
                for i in range(cnt, 200):
                    st = [0,0,0,0,0,0,0,0,0,0]
                    pt = [0,0,0,0,0,0,0,0]
                    tt = [0,0,0,0,0,0,0,0,0,0]
                    m = 0

                    cur_context.append([st, pt, tt, m])
                context.append(cur_context)
        return context

    def get_label(self, tag):
        with open(os.path.join(self.base_path, 'sample_' + tag + '_label.csv'), 'r') as f:
            p = f.readlines()
            p = [int(i)-1 for i in p]
        return p


if __name__ == "__main__":
    os.chdir(sys.path[0])
    dataTool = C_code()
    dataTool.preprocess_data()