import pandas as pd
import os
import sys
import numpy as np
import random


class C_code():
    def __init__(self):
        self.base_path = './data/java/'
        self.rawData_path = self.base_path + 'rawData'
        self.outData_path = self.base_path + 'outData'
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
            with open(os.path.join(self.tempData_path, str(id) + '.java'), 'w', encoding='utf-8') as f:
                f.write("class x { " + code + "}")

        # for root, dirs, files in os.walk(self.rawData_path):
        #    for f in files:
        cur = pd.read_csv('../java_codes.csv')
        cur.columns = ['unnamed', 'id', 'code']
        cur.apply(processLine, axis=1)

        ret = os.system(
            r'./astminer/cli.sh pathContexts --lang java --project ' + self.tempData_path + ' --output ' + self.outData_path +
            r' --maxL 8 --maxW 2 --maxContexts ' + str(200))

        assert ret == 0
        print("Extract Code Paths done!")
        
    def load_data(self, tag, t):
        path_dict = np.load('./data/java/path_dict.npy', allow_pickle=True).item()
        node_dict = np.load('./data/java/node_dict.npy', allow_pickle=True).item()
        context_x = []
        context_y = []
        f = pd.read_csv(os.path.join('./data/java/', tag + '.csv'))
        for index, row in f.iterrows():
            # if 0 <= row['label'] <= t:
            cur_context_x = []
            cur_context_y = []

            line = row['context_x']
            line = line.split(' ')
            line_2 = row['context_y']
            line_2 = line_2.split(' ')

            cnt = 0
            for path in line[1:]:
                source_token, path_token, target_token = map(int, path.strip().replace("\"", "").split(','))
                source_token = [int(i) for i in node_dict[source_token].split(",")]
                path_token = [int(i) for i in path_dict[path_token].split(",")]
                target_token = [int(i) for i in node_dict[target_token].split(",")]
                cur_context_x.append([source_token, path_token, target_token, 1])

                cnt = cnt + 1
            for i in range(cnt, 500):
                st = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pt = [0, 0, 0, 0, 0, 0, 0, 0]
                tt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                m = 0

                cur_context_x.append([st, pt, tt, m])
            context_x.append(cur_context_x)

            cnt_2 = 0
            for path in line_2[1:]:
                source_token, path_token, target_token = map(int, path.strip().replace("\"", "").split(','))
                source_token = [int(i) for i in node_dict[source_token].split(",")]
                path_token = [int(i) for i in path_dict[path_token].split(",")]
                target_token = [int(i) for i in node_dict[target_token].split(",")]
                cur_context_y.append([source_token, path_token, target_token, 1])

                cnt_2 = cnt_2 + 1
            for i in range(cnt_2, 500):
                st = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pt = [0, 0, 0, 0, 0, 0, 0, 0]
                tt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                m = 0

                cur_context_y.append([st, pt, tt, m])
            context_y.append(cur_context_y)

        return context_x, context_y

    def get_label(self, tag, t):
        # labels = []
        f = pd.read_csv(os.path.join('./data/c/', tag + '.csv'))
        return f['label'].values.tolist()

if __name__ == "__main__":
    os.chdir(sys.path[0])
    dataTool = C_code()
    dataTool.preprocess_data()

