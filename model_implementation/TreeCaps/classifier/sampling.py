"""Functions to help with sampling trees."""

import pickle
import numpy as np
import random
def gen_allsamples(alltrees, allabels,vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    ###
    allnode = []
    allchildren = []
    alllabel = []
    alllab =[]

    labels = list(set(allabels))
    labels = [int(i) for i in labels]
    list.sort(labels)
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}
    for tree in alltrees:
        nodes = []
        children = []
        label = label_lookup[int(tree['label'])]
        lab = int(tree['label'])-1
        queue = [(tree['tree'], -1)]
        while queue:
            node, parent_ind = queue.pop(0)
            node_ind = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            if node['node'] == 'root':
                node['node'] = 'FileAST'
            # if node['node'] == None:
            #     node['node'] = 'None'
            if node['node'] == 'FileAST':
                nodes.append(0)
            else:
                if node['node'] in vector_lookup:
                    nodes.append(vector_lookup[node['node']])
                else:
                    nodes.append(vector_lookup['UNK'])
        allnode.append(nodes)
        allchildren.append(children)
        alllab.append(lab)
        alllabel.append(label)

    return allnode,allchildren,alllab,alllabel

# def gen_samples(trees, labels, vectors, vector_lookup):
#     """Creates a generator that returns a tree in BFS order with each node
#     replaced by its vector embedding, and a child lookup table."""
#
#     # encode labels as one-hot vectors
#     ###
#     labels = list(set(labels))
#     label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}
#     for tree in trees:
#         nodes = []
#         children = []
#         label = label_lookup[tree['label']]
#         lab = int(tree['label'])-1
#         queue = [(tree['tree'], -1)]
#         while queue:
#             node, parent_ind = queue.pop(0)
#             node_ind = len(nodes)
#             # add children and the parent index to the queue
#             queue.extend([(child, node_ind) for child in node['children']])
#             # create a list to store this node's children indices
#             children.append([])
#             # add this child to its parent's child list
#             if parent_ind > -1:
#                 children[parent_ind].append(node_ind)
#             if node['node'] == 'root':
#                 node['node'] = 'FileAST'
#             # if node['node'] == None:
#             #     node['node'] = 'None'
#             if node['node'] == 'FileAST':
#                 nodes.append(vectors[0])
#             else :
#                 nodes.append(vectors[vector_lookup[node['node']]])
#
#         yield (nodes, children, label,lab)

def gen_samples(trees, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    ###
    labels = list(set(labels))
    labels = [int(i) for i in labels]
    list.sort(labels)
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}

    for tree in trees:
        nodes = []
        children = []
        label = label_lookup[int(tree['label'])]
        lab = int(tree['label'])-1
        queue = [(tree['tree'], -1)]
        while queue:
            node, parent_ind = queue.pop(0)
            node_ind = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            if node['node'] == 'root':
                node['node'] = 'FileAST'
            # if node['node'] == None:
            #     node['node'] = 'None'
            if node['node'] == 'FileAST':
                nodes.append(0)
            else :
                nodes.append(vector_lookup[node['node']])

        yield (nodes, children, label,lab)
def batch_onesample(allnode,allchildren,alllab,alllabel,ind, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels, one_lab = [], [], [],[]
    for i in range(ind,ind+batch_size):
        nodes.append(allnode[i])
        children.append(allchildren[i])
        labels.append(alllabel[i])
        one_lab.append(alllab[i])


    return _pad_batch(nodes, children, labels, one_lab)

def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels, one_lab = [], [], [],[]
    samples = 0
    for n, c, l, ll in gen:
        nodes.append(n)
        children.append(c)
        labels.append(l)
        one_lab.append(ll)
        samples += 1
        if samples >= batch_size:
            yield _pad_batch(nodes, children, labels,one_lab)
            nodes, children, labels,one_lab = [], [], [],[]
            samples = 0

    if nodes:
        yield _pad_batch(nodes, children, labels, one_lab)

# def _pad_batch(nodes, children, labels,one_lab):
#     if not nodes:
#         return [], [], []
#     max_nodes = max([len(x) for x in nodes])
#     max_children = max([len(x) for x in children])
#     feature_len = len(nodes[0][0])
#     child_len = max([len(c) for n in children for c in n])
#
#     nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
#     # pad batches so that every batch has the same number of nodes
#     children = [n + ([[]] * (max_children - len(n))) for n in children]
#     # pad every child sample so every node has the same number of children
#     children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]
#
#     return nodes, children, labels,one_lab


def _pad_batch(nodes, children, labels,lab):
    if not nodes:
        return [], [], [],[]
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [0] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children, labels,lab

def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]
