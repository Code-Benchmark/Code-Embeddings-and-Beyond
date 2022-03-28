from collections import defaultdict

import os

from pycparser import parse_file

def read_oj_scripts(data_dir):
    result = []
    label_counts = defaultdict(int)
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for script_index in os.listdir(label_dir):
            ast_tree = parse_file(os.path.join(label_dir, script_index), use_cpp=True)
            result.append({
                'tree': ast_tree, 'metadata': {'label': label}
            })
            label_counts[label] += 1
    print('Label counts:', label_counts)
    return result