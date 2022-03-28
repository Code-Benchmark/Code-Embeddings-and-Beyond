"""Parse nodes from a given data source."""

import pickle
from collections import defaultdict
from java_ast import JavaAST
import argparse
from tqdm import tqdm


def parse(args):
    """Parse nodes with the given args."""

    nodes_path = '../data/nodemap.pkl'
    print('Loading pickle file')

    with open(args.infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)
    print('Pickle load finished')

    node_counts = defaultdict(int)
    samples = []
    alltoken = []
    has_capacity = lambda x: args.per_node < 0 or node_counts[x] < args.per_node
    can_add_more = lambda: args.limit < 0 or len(samples) < args.limit

    for item in tqdm(data_source):
        root = item['tree']
        new_samples = [
            {
                'node': _name(root),
                'parent': None,
                # 'children': [_name(x) for x in ast.iter_child_nodes(root)]
                'children': [_name(x) for x in root.children()]
                # 'children': [x.name for x in root.child]
            }
        ]
        tokens = []
        gen_samples = lambda x: new_samples.extend(_create_samples(x))
        _traverse_tree(root, gen_samples)
        for sample in new_samples:
            if has_capacity(sample['node']):
                samples.append(sample)
                node_counts[sample['node']] += 1
                tokens.append(sample['node'])
            if not can_add_more:
                break
        alltoken.append(tokens)
        if not can_add_more:
            break
    print('dumping sample')

    with open(args.outfile, 'wb') as file_handler:
        pickle.dump(samples, file_handler)
        file_handler.close()

    print('Sampled node counts:')
    # print(node_counts)
    print('Copy the following list to vectorizer/node_map.py')
    # print(node_counts.keys())
    # path = '/tbcnn-data/searchdata/partmap.pkl'
    k = list(node_counts.keys())
    # k.append('FileAST')
    # with open(path, 'wb') as fo:
    #     pickle.dump(k, fo)

    with open(nodes_path, 'wb') as fo:
        pickle.dump(k, fo)
    print('Total: %d' % sum(node_counts.values()))


def _create_samples(node):
    """Convert a node's children into a sample points."""
    samples = []
    # for child in ast.iter_child_nodes(node):
    for _, child in node.children():
        sample = {
            "node": _name(child),
            "parent": _name(node),
            # "children": [_name(x) for x in ast.iter_child_nodes(child)]
            "children": [_name(x) for x in child.children()]
        }
        samples.append(sample)

    return samples


def _traverse_tree(tree, callback):
    """Traverse a tree and execute the callback on every node."""

    queue = [tree]
    while queue:
        current_node = queue.pop(0)
        children = list(child for _, child in current_node.children())
        # children = list(ast.iter_child_nodes(current_node))
        queue.extend(children)
        callback(current_node)


def _name(node):
    """Get the name of a node."""
    # if isinstance(node, JavaAST):
    if type(node).__name__ == 'JavaAST':
        return node.name
        # if isinstance(node, tuple):
        #     return node[1].name
        # else:
        #     return node.name
    else:
        return type(node).__name__


def main():
    parser = argparse.ArgumentParser(
        description="Crawl data sources for Python scripts.",
    )
    parser.add_argument('--infile', type=str, default='../data/example.pkl',
                        help='Data file to sample from')
    parser.add_argument('--outfile', type=str, default='../data/example_nodes.pkl',
                        help='File to store samples in')
    parser.add_argument(
        '--per-node', type=int, default=-1,
        help='Sample up to a maxmimum number for each node type'
    )
    parser.add_argument(
        '--limit', type=int, default=-1,
        help='Maximum number of samples to store.'
    )

    args = parser.parse_args()
    parse(args)


if __name__ == '__main__':
    main()