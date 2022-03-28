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
    # try:
    #     return torch.softmax(logits, axis=axis)
    # except:
    #     return torch.softmax(logits, dim=axis)

    return torch.nn.functional.softmax(logits, dim=axis)

# def get_shape(inputs, name=None):
#     name = "shape" if name is None else name
#     with tf.name_scope(name):
#         static_shape = inputs.get_shape().as_list()
#         dynamic_shape = tf.shape(inputs)
#         shape = []
#         for i, dim in enumerate(static_shape):
#             dim = dim if dim is not None else dynamic_shape[i]
#             shape.append(dim)
#         return(shape)