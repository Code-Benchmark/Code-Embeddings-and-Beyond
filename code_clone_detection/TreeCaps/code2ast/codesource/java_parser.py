import json
import pickle
from code2ast.codesource.java_ast import JavaAST
from javalang.ast import Node
import javalang
import pandas as pd


def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

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

def get_sequence(node,sequence):

    token, children = get_token(node), get_children(node)
    ast_node = JavaAST(token, value='', child=[])

    sequence.append(token)
    for child in children:
        # if not isinstance(node, str):
        ast_child = get_sequence(child, sequence)
        if ast_child is not None:
            ast_node.child.append(ast_child)

    return ast_node

def read_java_json(infile, outfile=None):
    data = []
    source = pd.read_csv(infile)
    source.columns = ['id', 'code']
    for i in range(len(source['id'])):
        sequence = []
        ast = parse_program(source['code'][i])
        data.append({
            'tree': get_sequence(ast,sequence), 'id':  source['id'][i]
        })


    if outfile is not None:
        print('Dumping scripts')
        with open(outfile, 'wb') as file_handler:
            pickle.dump(data, file_handler, -1)

    return data
