import pandas as pd
import os
import sys
import numpy as np
import torch
import random


class C_code():
    def __init__(self):
        self.base_path = './data/data/'
        self.rawData_path = self.base_path + 'rawData'
        self.outData_path = self.base_path + 'outData_all'
        self.tempData_path = self.base_path + 'tempData'

    # execute for the first time
    def preprocess_data(self):
        if not os.path.exists(self.outData_path):
            os.mkdir(self.outData_path)
        if not os.path.exists(self.tempData_path):
            os.mkdir(self.tempData_path)

        def processLine(row):
            id = row['index']
            code = row['code']
            with open(os.path.join(self.tempData_path, str(id) + '.java'), 'w', encoding='utf-8') as f:
                f.write("class TEST { " + code + "}")

        # for root, dirs, files in os.walk(self.rawData_path):
        #    for f in files:
        cur = pd.read_csv(os.path.join(self.base_path, 'data.csv'))
        cur.columns = ['id', 'index', 'code', 'docstring', 'partition', 'docstring_clean']
        # cur.apply(processLine, axis=1)

        # ret = os.system(
        #     r'java -jar cli.jar pathContexts --lang c --project ' + self.tempData_path + ' --output ' + self.outData_path +
        #     r' --maxH 8 --maxW 2 --maxContexts ' + str(200) + ' --maxTokens ' + str(1301136) +
        #     ' --maxPaths ' + str(911417))
        ret = os.system(
            r'./astminer/cli.sh pathContexts --lang java --project ' + self.tempData_path + ' --output ' + self.outData_path )#+
            # r' --maxL 8 --maxW 2')#--maxContexts  + str(200) + ' --maxTokens ' + str(
                # 1000000) + ' --maxPaths ' + str(1000000))

        assert ret == 0
        print("Extract Code Paths done!")

    def load_data(self, tag):
        token_dict = np.load('./data/new_token_id_dict.npy', allow_pickle=True).item()
        # node_dict = np.load('./data/node_dict.npy').item()
        context = []
        with open(os.path.join('./data/', tag + '_code_0304.csv'), 'r', encoding='UTF-8') as f:

            for line in f:
                if line == "context\n" or line == '"\n':
                    continue
                line = line.split(' ')

                cur_context = []

                cnt = 0
                for path in line[1:]:
                    source_token, path_token, target_token = map(int, path.strip().replace("\"", "").split(','))
                    source_token = token_dict[source_token]
                    target_token = token_dict[target_token]
                    cur_context.append([source_token, path_token, target_token, 1])

                    cnt = cnt + 1
                for i in range(cnt, 500):
                    st = 0
                    pt = 0
                    tt = 0
                    m = 0
                    cur_context.append([st, pt, tt, m])
                context.append(cur_context)

        return context

    def get_label(self, tag):
        labels = []
        f = pd.read_csv(os.path.join('./data/java/', tag + '_data.csv'))

        return labels


if __name__ == "__main__":
    os.chdir(sys.path[0])
    dataTool = C_code()
    dataTool.preprocess_data()
