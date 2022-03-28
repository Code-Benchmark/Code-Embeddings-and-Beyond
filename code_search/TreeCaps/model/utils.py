from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn import preprocessing
import os, itertools
import pickle
import random
import torch
import scipy

def reduce_sum(input_tensor, axis=None, keepdims=False):

        return torch.sum(input_tensor, dim=axis, keepdims=keepdims)



# For version compatibility
def softmax(logits, axis=None):
    return torch.nn.functional.softmax(logits, dim=axis)

