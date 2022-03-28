"""Parse trees from a data source."""
import pickle
import random
import sys
from collections import defaultdict
import pandas as pd
import os
from java_ast import JavaAST
from tqdm import tqdm
from javalang.ast import Node
import argparse


def parse(args):
    """Parse trees with the given arguments."""

    print("splitting data...")
    if args.lang =='c':
        split_data(args.pairspath, args.filepath, args.lang, args.ratio)

    print('Loading ast pickle file')

    sys.setrecursionlimit(1000000)
    print(args.infile)
    path = r'..\data\example.pkl'

    # with open(args.infile, 'rb') as file_handler:
    # with open(path, 'rb') as file_handler:
    #     data_source = pickle.load(file_handler)
    data_source = pickle.load(open(path, 'rb'))

    print('Pickle file load finished')

    data_samples = []
    # print data_source
    for item in tqdm(data_source):
        root = item['tree']
        id = item['id']
        if args.lang == 'c':
            sample, size = _traverse_tree(root)
        elif args.lang == 'java':
            sample, size = _traverse_java_tree(root)

        if size > args.maxsize or size < args.minsize:
            print('continue', size, args.maxsize, args.minsize)
            continue

        roll = random.randint(0, 100)

        datum = {'tree': sample, 'id': id}

        data_samples.append(datum)

    print('Dumping sample')
    with open(args.outfile, 'wb') as file_handler:
        pickle.dump((data_samples), file_handler)
        file_handler.close()
    print('dump finished')

def split_data(pairspath,filepath,lang,ratio):
    data_path = filepath
    data = pd.read_pickle(pairspath)

    data_num = len(data)
    ratios = [int(r) for r in ratio.split(':')]
    train_split = int(ratios[0] / sum(ratios) * data_num)
    val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

    data = data.sample(frac=1, random_state=666)
    train = data.iloc[:train_split]
    dev = data.iloc[train_split:val_split]
    test = data.iloc[val_split:]

    def check_or_create(path):
        if not os.path.exists(path):
            os.mkdir(path)

    train_path = data_path + 'train/'
    check_or_create(train_path)
    train_file_path = train_path + 'train_.pkl'
    train.to_pickle(train_file_path)

    dev_path = data_path + 'dev/'
    check_or_create(dev_path)
    dev_file_path = dev_path + 'dev_.pkl'
    dev.to_pickle(dev_file_path)

    test_path = data_path + 'test/'
    check_or_create(test_path)
    test_file_path = test_path + 'test_.pkl'
    test.to_pickle(test_file_path)


def _traverse_java_tree(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": root.name,

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        delete_c = JavaAST(name='', value='', child=[])
        for _, child in current_node.children():
            if child.name != 'Token':
                delete_c.child.append(child)

        num_nodes += 1
        # print (_name(current_node))

        current_node_json = queue_json.pop(0)

        children = list(child for _, child in delete_c.children())

        queue.extend(children)
        for child in children:
            child_json = {
                # "node": _name(child),
                "node": child.name,
                "children": []
            }

            current_node_json['children'].append(child_json)
            # if child.name != 'Token':
            queue_json.append(child_json)

    return root_json, num_nodes

def _traverse_tree(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)

        children = list(child for _, child in current_node.children())
        # children = list(ast.iter_child_nodes(current_node))
        queue.extend(children)
        for child in children:
            child_json = {
                "node": _name(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes

def _name(node):

    if isinstance(node, JavaAST):
        return node.name
    else:
        return node.__class__.__name__

def get_token(node):
    token = ''
    if isinstance(node, str):
        # token = node
        token = 'Token'
    elif isinstance(node, set):
        token = 'Modifier'#node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token


def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))

def main():

    parser = argparse.ArgumentParser(
        description="Crawl data sources for Python scripts.",
    )
    parser.add_argument('--infile', type=str, default='../data/example.pkl',
                             help='Data file to sample from')
    parser.add_argument('--outfile', type=str, default='../data/example_trees.pkl',
                             help='File to store samples in')
    parser.add_argument('--ratio', default='8:1:1', type=str, help='Percent to save as test data')

    parser.add_argument('--pairspath', type=str, default='../data/javanew', help='')
    parser.add_argument('--filepath', default='../data/clonedata/', type=str, help='')
    parser.add_argument('--lang', default='java', type=str, help='choose language')

    parser.add_argument(
        '--label-key', type=str, default='label',
        help='Change which key to use for the label'
    )
    parser.add_argument(
        '--maxsize', type=int, default=10000,
        help='Ignore trees with more than --maxsize nodes'
    )
    parser.add_argument(
        '--minsize', type=int, default=1,
        help='Ignore trees with less than --minsize nodes'
    )


    args = parser.parse_args()
    parse(args)
if __name__ =='__main__':
    main()