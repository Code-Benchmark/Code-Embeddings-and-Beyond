import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model_for_attr import BatchProgramCC
from torch.autograd import Variable
from captum.attr import IntegratedGradients

# Load Data
root = 'data/'
data_test = pd.read_pickle(root+'test/blocks.pkl')
word2vec = Word2Vec.load(root + "train/embedding/node_w2v_128").wv
MAX_TOKENS = word2vec.syn0.shape[0]
embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

# Param Setting
HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 104
EPOCHS = 15
BATCH_SIZE = 64
USE_GPU = True
MAX_TOKENS = word2vec.syn0.shape[0]
EMBEDDING_DIM = word2vec.syn0.shape[1]

# Load Model
model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                               USE_GPU, embeddings)
PATH = './model/model.pkl'
checkpoint = torch.load(PATH)
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
model.cuda()

# Test id
id = 32230
label = 56
test = data_test[data_test["id"]==id]

# Generate Input Embedding
input_origin = test["code"].values[0]
encodes_x = []
lens_x = len(input_origin)
for j in range(lens_x):
    encodes_x.append(input_origin[j])
input_indices = model.encoder(encodes_x, lens_x)
input_indices = input_indices.unsqueeze(0)
print("Test: " + str(model(input_indices).argmax()+1))

# Baseline
reference_indices = torch.zeros_like(input_indices)

# Compute IG 
lig = IntegratedGradients(model)
attributions_ig, delta = lig.attribute(input_indices, reference_indices, target=label-1, return_convergence_delta=True)
print("delta: " + str(delta))
attr_score = attributions_ig.squeeze(0).abs().sum(1).cpu().detach().numpy()

# Print Result
for i in range(0, lens_x):
    print(str(input_origin[i]) + ' : ' + str(attr_score[i]))

