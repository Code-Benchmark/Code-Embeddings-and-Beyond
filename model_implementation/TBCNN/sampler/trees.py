"""Parse trees from a data source."""
import pickle
import random
import sys
from collections import defaultdict

from java_ast import JavaAST

def parse(inputfilepath, outputfilepath):
    """Parse trees with the given arguments."""
    print('Loading pickle file')

    sys.setrecursionlimit(1000000)
    print(inputfilepath)
    with open(inputfilepath, 'rb') as file_handler:
        data_source = pickle.load(file_handler)

    print('Pickle file load finished')

    train_samples = []
    dev_samples = []
    test_samples = []

    train_counts = defaultdict(int)
    dev_counts = defaultdict(int)
    test_counts = defaultdict(int)
    ratio = '3:1:1'
    ratio =[int(r) for r in ratio.split(':')]
    train_r = 100 * ratio[0]/sum(ratio)
    dev_r = 100 * (ratio[1]+ratio[0]) / sum(ratio)
    # print data_source
    for item in data_source:
        # print(item)
        root = item['tree']
        label = item['metadata']['label']
        sample, size = _traverse_tree(root)

        # if size > args.maxsize or size < args.minsize:
        #     print('continue', size, args.maxsize, args.minsize)
        #     continue

        roll = random.randint(0, 100)

        datum = {'tree': sample, 'label': label}

        if roll < train_r:
            train_samples.append(datum)
            train_counts[label] += 1
        elif roll >= train_r and roll < dev_r:
            dev_samples.append(datum)
            dev_counts[label] += 1
        else:
            test_samples.append(datum)
            test_counts[label] += 1


    random.shuffle(train_samples)
    random.shuffle(test_samples)
    random.shuffle(dev_samples)
    # create a list of unique labels in the data
    # labels = list(set(train_counts.keys() + test_counts.keys()))
    labels = list(train_counts.keys()) + list(test_counts.keys()) + list(dev_counts.keys())
    print('Dumping sample')
    with open(outputfilepath, 'wb') as file_handler:
        pickle.dump((train_samples, dev_samples, test_samples, labels), file_handler)
        file_handler.close()
    print('dump finished')
    print('Sampled tree counts: ')
    print('Training:', train_counts)
    print('Testing:', test_counts)

def _traverse_tree(root):
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _nodes_name(root),

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
                "node": _nodes_name(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes

def _name(node):
    # if type(node).__name__=="FileAST":
    #     return "root"
    # return node.__class__.__name__
    if isinstance(node, JavaAST):
        return node.name
    else:
        return node.__class__.__name__

def _nodes_name(node):
    """Get the name of a node."""
    if isinstance(node, JavaAST):
        if isinstance(node, tuple):
            return node[1].name
        else:
             # if node.children() == None:

            return node.name
    else:
        if node.children() == ():
            if hasattr(node, 'name'):
                return node.name
            elif hasattr(node, 'type'):
                return node.type
        else:
            return type(node).__name__

if __name__ == "__main__":
    inputfilepath = '/tbcnn-data/poj/poj.pkl'
    outputfilepath = '/tbcnn-data/poj/poj_trees.pkl'
    parse(inputfilepath, outputfilepath)