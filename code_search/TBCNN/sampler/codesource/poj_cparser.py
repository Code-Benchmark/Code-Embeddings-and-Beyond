from collections import defaultdict
import pandas as pd
import os

from pycparser import parse_file
from pycparser import c_parser
# def read_oj_scripts(data_dir):
#
#     result = []
#     label_counts = defaultdict(int)
#     for label in os.listdir(data_dir):
#         label_dir = os.path.join(data_dir, label)
#         for script_index in os.listdir(label_dir):
#             ast_tree = parse_file(os.path.join(label_dir, script_index), use_cpp=True)
#             result.append({
#                 'tree': ast_tree, 'metadata': {'label': label}
#             })
#             label_counts[label] += 1
#     print('Label counts:', label_counts)
#     return result

def read_oj_scripts(data_dir):
    result = []
    label_counts = defaultdict(int)
    source = pd.read_pickle(data_dir)
    source.columns = ['id', 'code', 'label']
    parser = c_parser.CParser()
    for i in range(len(source)):
        ast_tree = parser.parse(source['code'][i])
        result.append({
            'id': source['id'][i], 'tree': ast_tree, 'metadata': {'label': source['label'][i]}
        })
        label_counts[source['label'][i]] += 1
    print('Label counts:', label_counts)
    return result

    # for label in os.listdir(data_dir):
    #     label_dir = os.path.join(data_dir, label)
    #     for script_index in os.listdir(label_dir):
    #         ast_tree = parse_file(os.path.join(label_dir, script_index), use_cpp=True)
    #         result.append({
    #             'tree': ast_tree, 'metadata': {'label': label}
    #         })
    #         label_counts[label] += 1
    # print('Label counts:', label_counts)
    # return result