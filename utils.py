import copy
import math
from collections import Counter

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy.api as sc
import scipy as sp
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def run_leiden(data, n_neighbors=10, n_pcs=40):
    import scanpy.api as sc
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X')
    sc.tl.leiden(adata)
    pred = adata.obs['leiden'].to_list()
    pred = [int(x) for x in pred]
    return pred

def augment(inp, zeros, nb_zeros = 3, perc = 0.1, random = False, augm_value = 0):
    aug = inp.copy()
    #np.mean(inp)
    random_vec = None
    if random:
        random_vec = np.random.normal(0, 0.01, size = inp.shape[1])
    for i in range(len(aug)):
        zero_idx = np.random.choice(np.arange(inp.shape[1]), nb_zeros, replace= False)
        if zeros is not None:
            if perc is None:
                aug[i, zero_idx] = zeros[zero_idx]
            else:
                sel_idx = np.arange(nb_zeros)[:int(nb_zeros * perc)]
                aug[i, zero_idx[sel_idx]] = zeros[zero_idx[sel_idx]]
                sel_idx = np.arange(nb_zeros)[int(nb_zeros * perc):]
                aug[i, zero_idx[sel_idx]] = augm_value
        elif random_vec is not None:
            aug[i, zero_idx] = aug[i, zero_idx] + random_vec[zero_idx]
        else:
            aug[i, zero_idx] = augm_value
    return aug


def evaluate(data, Yt, cluster_number):
    kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0)
    kmeans_pred = kmeans.fit_predict(data)
#     cm = get_coassociation_matrix(kmeans_pred)

#     if epoch > 8:
#         cm = momentum * prev_cm + (1 - momentum) * cm

    kmeans_accuracy = np.around(cluster_acc(Yt, kmeans_pred), 5)
    kmeans_ARI = np.around(adjusted_rand_score(Yt, kmeans_pred), 5)
    kmeans_NMI = np.around(
        normalized_mutual_info_score(Yt, kmeans_pred), 5)
    return kmeans_accuracy,kmeans_ARI, kmeans_NMI


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
