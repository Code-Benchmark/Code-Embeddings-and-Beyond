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
        cur = pd.read_csv('../codes.csv')
        cur.columns = ['unnamed', 'id', 'code']
        cur.apply(processLine, axis=1)

        ret = os.system(
            r'./astminer/cli.sh pathContexts --lang java --project ' + self.tempData_path + ' --output ' + self.outData_path +
            r' --maxL 8 --maxW 2 --maxContexts ' + str(200))

        assert ret == 0
        print("Extract Code Paths done!")




    def load_data(self, tag, t):
        source_tokens = []
        path_tokens = []
        target_tokens = []
        context_valid_masks = []
        source_tokens_2 = []
        path_tokens_2 = []
        target_tokens_2 = []
        context_valid_masks_2 = []

        f = pd.read_csv(os.path.join('./data/java/', tag + '_data.csv'))
        for index, row in f.iterrows():
            # if 0 <= row['label'] <= t:
            cur_source_tokens = []
            cur_path_tokens = []
            cur_target_tokens = []
            cur_context_masks = []

            cur_source_tokens_2 = []
            cur_path_tokens_2 = []
            cur_target_tokens_2 = []
            cur_context_masks_2 = []

            line = row['context_x']
            line = line.split(' ')
            line_2 = row['context_y']
            line_2 = line_2.split(' ')

            cnt = 0
            for path in line[1:]:
                source_token, path_token, target_token = map(int, path.strip().split(','))

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

            cnt_2 = 0
            for path in line_2[1:]:
                source_token_2, path_token_2, target_token_2 = map(int, path.strip().split(','))

                cur_source_tokens_2.append(source_token_2)
                cur_path_tokens_2.append(path_token_2)
                cur_target_tokens_2.append(target_token_2)
                cur_context_masks_2.append(1)
                cnt_2 = cnt_2 + 1

            cur_source_tokens_2 = np.pad(cur_source_tokens_2, (0, 200 - cnt_2), 'constant',
                                         constant_values=(0, 0))
            cur_path_tokens_2 = np.pad(cur_path_tokens_2, (0, 200 - cnt_2), 'constant',
                                       constant_values=(0, 0))
            cur_target_tokens_2 = np.pad(cur_target_tokens_2, (0, 200 - cnt_2), 'constant',
                                         constant_values=(0, 0))
            cur_context_masks_2 = np.pad(cur_context_masks_2, (0, 200 - cnt_2), 'constant',
                                         constant_values=(0, 0))

            source_tokens_2.append(cur_source_tokens_2)
            path_tokens_2.append(cur_path_tokens_2)
            target_tokens_2.append(cur_target_tokens_2)
            context_valid_masks_2.append(cur_context_masks_2)

        return source_tokens, path_tokens, target_tokens, context_valid_masks \
            , source_tokens_2, path_tokens_2, target_tokens_2, context_valid_masks_2

    def get_label(self, tag, t):
        f = pd.read_csv(os.path.join('./data/c/', tag + '.csv'))
        if tag != 'test':
            f.loc[f['label'] > 0, 'label'] = 1

        return f['label'].values.tolist()




if __name__ == "__main__":
    os.chdir(sys.path[0])
    dataTool = C_code()
    dataTool.preprocess_data()

