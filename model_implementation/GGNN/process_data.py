#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
import pickle
import os
import json


# In[38]:


dict_file = 'data/java_clone/map_dir/maps.java.pkl'


# In[39]:


vocab_data = pickle.load(open(dict_file,'rb'))


# In[40]:


len(vocab_data)


# In[41]:


vocab_data['CALL']


# In[42]:


id2key = {int(vocab_data[key]):key for key in vocab_data.keys()}


# In[43]:


id2key[2]


# In[44]:


node_types = [key.split(',')[0] for key in vocab_data.keys()]


# In[45]:


node_freq_dict = {}
for types in node_types:
    node_freq_dict[types] = node_freq_dict.get(types,0)+1


# In[46]:


node_freq_dict


# In[47]:


len(node_freq_dict)


# In[48]:


nodetype2id_map = {item[0]:idx for idx,item in enumerate(sorted(list(node_freq_dict.items()),key=lambda x:x[1],reverse=True))}


# In[49]:


nodetype2id_map


# In[50]:


data_dir = "../data/java_clone/"
# tasks = ['train','dev','test']

node_dict = {}

# for task in ['train', 'dev', 'test']:
for root, dirs, files in os.walk('data/java_clone/graph'):
    for filename in files:
        if filename.endswith('.txt'):
            with open(os.path.join(root, filename)) as f:
                lines = f.readlines()
                for line in lines:
                    start_node,edge_type,end_node = line.strip().split(" ")
                    start_node = int(start_node)
                    end_node = int(end_node)
                    node_dict[start_node] = node_dict.get(start_node,0) + 1
                    node_dict[end_node] = node_dict.get(start_node,0) + 1


# In[51]:


sorted_nodes = sorted(list(node_dict.items()),key=lambda x: x[1], reverse=True)


# In[52]:


sorted_nodes[:10]


# In[53]:


new_node_map = {item[0]:idx for idx,item in enumerate(sorted_nodes)}


# In[54]:


reverse_node_map = {idx:item[0] for idx,item in enumerate(sorted_nodes)}


# In[55]:


new_node_map[14]


# In[56]:


new_graph_data = {}

#for task in ['train', 'dev', 'test']:
# new_graph_data[task] = {}
for root, dirs, files in os.walk('data/java_clone/graph'):
    for filename in files:
        if filename.endswith('.txt'):
            line_no = int(filename.split('.')[0])
            new_graph_data[line_no] = {}
            with open(os.path.join(root, filename)) as f:
                lines = f.readlines()
                new_lines = []
                nodes = set()
                for line in lines:
                    start_node,edge_type,end_node = line.strip().split(" ")
                    start_node = int(start_node)
                    end_node = int(end_node)
                    new_start_node = new_node_map[start_node]
                    new_end_node = new_node_map[end_node]
                    new_lines.append('%d %s %d'%(new_start_node,edge_type,new_end_node))
                    nodes.add(new_start_node)
                    nodes.add(new_end_node)
                sort_nodes = sorted(list(nodes))
                nodes_map = {node_id:idx for idx,node_id in enumerate(sort_nodes)}
                start_edges = []
                end_edges = []
                edge_types = []
                for line in new_lines:
                    start_node,edge_type,end_node = line.strip().split(" ")
                    start_node = int(start_node)
                    end_node = int(end_node)
                    edge_type = int(edge_type)
                    start_edges.append(nodes_map[start_node])
                    end_edges.append(nodes_map[end_node])
                    edge_types.append(edge_type)
                node_types = []
                node_texts = []
                for node_id in sort_nodes:
                    vocab_key = id2key[reverse_node_map[node_id]]
                    if ',' in vocab_key:
                        node_type = vocab_key.split(',')[0]
                        text = vocab_key[len(node_type)+1:][1:-1].strip()
                        node_type = nodetype2id_map[node_type]
                    else:
                        node_type = nodetype2id_map[vocab_key]
                        text = vocab_key
                    node_types.append(node_type)
                    node_texts.append(text)
                new_graph_data[line_no]['node_ids'] = sort_nodes
                new_graph_data[line_no]['edges'] = [start_edges,end_edges]
                new_graph_data[line_no]['node_types'] = node_types
                new_graph_data[line_no]['node_texts'] = node_texts
                new_graph_data[line_no]['edge_types'] = edge_types


# In[57]:


for key in new_graph_data.keys():
    print(key)
    print(new_graph_data[2])
    print(len(new_graph_data[0]['edges'][0]))
    nodes = new_graph_data[0]['node_ids']
    edge_types = new_graph_data[0]['edge_types']
    start_nodes = new_graph_data[0]['edges'][0]
    end_nodes = new_graph_data[0]['edges'][1]
    for idx,(s,e) in enumerate(zip(start_nodes,end_nodes)):
        row_s = reverse_node_map[nodes[s]]
        edge_type = edge_types[idx]
        row_e = reverse_node_map[nodes[e]]
        print('%d %d %d'%(row_s,edge_type,row_e))
    for nid in nodes:
        row_key = id2key[reverse_node_map[nid]]
        print('%d %d %s'%(nid,reverse_node_map[nid],row_key))
    break


# In[59]:


with open('data/java_clone/ggnn.json','w') as f:
    task_json = new_graph_data
    sort_json = {key:value for key,value in sorted(list(task_json.items()),key=lambda x:x[0])}
    f.write(json.dumps(sort_json))






