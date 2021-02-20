"""
Original implementation of Contrastive-sc method
(https://github.com/ciortanmadalina/contrastive-sc)
By Madalina Ciortan (01/10/2020)
"""
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


def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, 
              size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalizes input data and retains only most variable genes 
    (indicated by highly_genes parameter)

    Args:
        adata ([type]): [description]
        copy (bool, optional): [description]. Defaults to True.
        highly_genes ([type], optional): [description]. Defaults to None.
        filter_min_counts (bool, optional): [description]. Defaults to True.
        size_factors (bool, optional): [description]. Defaults to True.
        normalize_input (bool, optional): [description]. Defaults to True.
        logtrans_input (bool, optional): [description]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
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
        sc.pp.filter_genes(adata, min_counts=1)#3
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
    """
    Computes clustering accuracy.
    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def run_leiden(data, leiden_n_neighbors=300):
    """
    Performs Leiden community detection on given data.

    Args:
        data ([type]): [description]
        n_neighbors (int, optional): [description]. Defaults to 10.
        n_pcs (int, optional): [description]. Defaults to 40.

    Returns:
        [type]: [description]
    """
    import scanpy.api as sc
    n_pcs=0
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, n_neighbors=leiden_n_neighbors, n_pcs=n_pcs, use_rep='X')
    sc.tl.leiden(adata)
    pred = adata.obs['leiden'].to_list()
    pred = [int(x) for x in pred]
    return pred

def augment(inp, zeros, nb_zeros = 3, perc = 0.1, random = False, augm_value = 0):
    """
    Handles the cell-level data augmentation which is primarily based on 
    input dropout.

    Args:
        inp ([type]): [description]
        zeros ([type]): [description]
        nb_zeros (int, optional): [description]. Defaults to 3.
        perc (float, optional): [description]. Defaults to 0.1.
        random (bool, optional): [description]. Defaults to False.
        augm_value (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
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
    """
    Performs K-means on input data with cluster_number clusters and evaluates 
    the result by computing the ARI, NMI and clustering accuracy 
    against the provided ground truth.

    Args:
        data ([type]): [description]
        Yt ([type]): [description]
        cluster_number ([type]): [description]

    Returns:
        [type]: [description]
    """
    kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0)
    kmeans_pred = kmeans.fit_predict(data)

    kmeans_accuracy = np.around(cluster_acc(Yt, kmeans_pred), 5)
    kmeans_ARI = np.around(adjusted_rand_score(Yt, kmeans_pred), 5)
    kmeans_NMI = np.around(
        normalized_mutual_info_score(Yt, kmeans_pred), 5)
    return kmeans_accuracy,kmeans_ARI, kmeans_NMI


def adjust_learning_rate(p, optimizer, epoch):
    """
    Adapts optimizer's learning rate during the training. 

    Args:
        p ([type]): [description]
        optimizer ([type]): [description]
        epoch ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
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

def rename_column(x, scenario = "ours"):
    ours_basic = [
        'method1_0',
        'method1_1',
        'method1_2',
    ]
    ours_clust = ['method2_0', 'method2_1', 'method2_2']
    if x in ours_basic:
        return "Baseline k-means"
    if x in ours_clust:
        return "Cluster NN"
    if x == 'method3':
        return "Ensemble K-means"
    if x == 'Leiden':
        if scenario == "ours":
            return "Ensemble Leiden (contrastive-sc)"
        else:
            return "contrastive-sc"
    if x == "pca":
        return "PCA + KMeans"
    if x == "original":
        return "KMeans"
    return x

def order_column(x):
    if x == "PCA + KMeans":
        return 0
    if x == "KMeans":
        return 1
    if x == 'Seurat':
        return 3
    if x == 'scanpy':
        return 4
    if x == 'sczi':
        return 5
    if x == "scDeepCluster":
        return 6
    if x == "Baseline k-means":
        return 7
    if x == "Cluster NN":
        return 8
    if x == "Ensemble K-means":
        return 9
    if x == "contrastive-sc":
        return 10
    if x == "Ensemble Leiden (contrastive-sc)":
        return 10

    return x
