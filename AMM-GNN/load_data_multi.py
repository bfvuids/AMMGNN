import scipy.io as scio
import numpy as np
import torch
from torch.nn import functional as F
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from sklearn.metrics import f1_score

def preprocess_citation(adj, features, normalization="AugNormAdj"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data_multi(dataFile='dblp'):

    data = scio.loadmat(dataFile)
    adj = data['Network']
    feas = data['Attributes']
    labels = data['Label']
    adj, feas = preprocess_citation(adj, feas, normalization="AugNormAdj")
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    feas = torch.FloatTensor(np.array(feas.todense())).float()
    labels = torch.tensor(labels).reshape(labels.shape[0])
    return adj, feas, labels
