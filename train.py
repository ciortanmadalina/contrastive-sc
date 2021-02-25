"""
Original implementation of Contrastive-sc method
(https://github.com/ciortanmadalina/contrastive-sc)
By Madalina Ciortan (01/10/2020)
"""
import argparse
import copy
import math
import os
import pickle
import time
import warnings
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
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             normalized_mutual_info_score, silhouette_score)

import models
import st_loss
import train
import utils

warnings.filterwarnings("ignore", category=FutureWarning)


def preprocess(X, nb_genes = 500):
    """
    Preprocessing phase as proposed in scanpy package.
    Keeps only nb_genes most variable genes and normalizes
    the data to 0 mean and 1 std.
    Args:
        X ([type]): [description]
        nb_genes (int, optional): [description]. Defaults to 500.
    Returns:
        [type]: [description]
    """
    X = np.ceil(X).astype(np.int)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
    orig_X = X.copy()
    adata = sc.AnnData(X)

    adata = utils.normalize(adata,
                      copy=True,
                      highly_genes=nb_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X = adata.X.astype(np.float32)
    return X

def adjust_learning_rate( optimizer, epoch, lr):
    p = {
      'epochs': 500,
     'optimizer': 'sgd',
     'optimizer_kwargs': {'nesterov': False,
              'weight_decay': 0.0001,
              'momentum': 0.9,

                         },
     'scheduler': 'cosine',
     'scheduler_kwargs': {'lr_decay_rate': 0.1},
     }

    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


def get_device(use_cpu):
    """[summary]

    Returns:
        [type]: [description]
    """
    if use_cpu is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    return device

def train_model(X,
                cluster_number,
                Y=None,
                nb_epochs=30,
                lr=0.4,
                temperature=0.07,
                dropout=0.9,
                evaluate_training = False,
                layers = [200, 40, 60],
                save_pred = False,
                noise = None,
                use_cpu = None):
    """[summary]

    Args:
        X ([type]): [description]
        cluster_number ([type]): [description]
        Y ([type], optional): [description]. Defaults to None.
        nb_epochs (int, optional): [description]. Defaults to 20.
        lr ([type], optional): [description]. Defaults to 1e-5.
        temperature (float, optional): [description]. Defaults to 0.07.
        dropout (float, optional): [description]. Defaults to 0.8.
        evaluate_training (bool, optional): [description]. Defaults to False.
        layers (list, optional): [description]. Defaults to [256, 64, 32].
        save_pred (bool, optional): [description]. Defaults to False.
        noise ([type], optional): [description]. Defaults to None.
        use_cpu ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    device = get_device(use_cpu)

    dims = np.concatenate([[X.shape[1]], layers])#[X.shape[1], 256, 64, 32]
    model = models.ContrastiveRepresentation(dims, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=lr)

    criterion_rep = st_loss.SupConLoss(temperature=temperature)
    batch_size = 200

    losses = []
    idx = np.arange(len(X))
    for epoch in range(nb_epochs):

        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_ = 0
        for pre_index in range(len(X) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(X), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = X[c_idx]
            if noise is None or noise ==0:
                input1 = torch.FloatTensor(c_inp).to(device)
                input2 = torch.FloatTensor(c_inp).to(device)
            else:
                noise_vec = np.random.normal(loc = 0, scale = noise, size = c_inp.shape)
                input1 = torch.FloatTensor(c_inp + noise_vec).to(device)
                noise_vec = np.random.normal(loc = 0, scale = noise, size = c_inp.shape)
                input2 = torch.FloatTensor(c_inp + noise_vec).to(device)


            anchors_output = model(input1)
            neighbors_output = model(input2)

            features = torch.cat(
                [anchors_output.unsqueeze(1),
                 neighbors_output.unsqueeze(1)],
                dim=1)
            total_loss = criterion_rep(features)
            loss_ += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        if evaluate_training and Y is not None:
            model.eval()
            with torch.no_grad():
                result = model(torch.FloatTensor(X))
                features = result.detach().cpu().numpy()
            res = train.cluster_embedding([features], cluster_number, Y, save_pred = save_pred)
            print(
                f"{epoch}). Loss {loss_}, ARI {res['kmeans_ari']}, {res['leiden_ari']}"
            )

        losses.append(loss_)
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(X).to(device))
        features = result.detach().cpu().numpy()
    return features



def run(X,
        cluster_number,
        dataset,
        Y=None,
        nb_epochs=30,
        lr=0.4,
        temperature=0.07,
        dropout=0.9,
        layers = [200, 40, 60],
        save_to="data/",
        save_pred = False,
        noise = None,
        use_cpu = None,
        cluster_methods = ["KMeans", "Leiden"],
        evaluate_training = False,
        leiden_n_neighbors=300):
    """[summary]

    Args:
        X ([type]): [description]
        cluster_number ([type]): [description]
        dataset ([type]): [description]
        Y ([type], optional): [description]. Defaults to None.
        nb_epochs (int, optional): [description]. Defaults to 30.
        lr (float, optional): [description]. Defaults to 0.4.
        temperature (float, optional): [description]. Defaults to 0.07.
        dropout (float, optional): [description]. Defaults to 0.9.
        layers (list, optional): [description]. Defaults to [256, 64, 32].
        save_to (str, optional): [description]. Defaults to "data/".
        save_pred (bool, optional): [description]. Defaults to False.
        noise ([type], optional): [description]. Defaults to None.
        use_cpu ([type], optional): [description]. Defaults to None.
        evaluate_training (bool, optional): [description]. Defaults to False.
        leiden_n_neighbors (int, optional): [description]. Defaults to 300.
    """
    results = {}

    start = time.time()
    embedding = train.train_model(X,
              cluster_number,
              Y=Y,
              nb_epochs=nb_epochs,
              lr=lr,
              temperature=temperature,
              dropout=dropout,
              layers = layers,
              evaluate_training=evaluate_training,
              save_pred= save_pred,
              noise = noise, 
              use_cpu = use_cpu)
    if save_pred:
        results[f"features"] = embedding
    elapsed = time.time() -start
    res_eval = train.cluster_embedding(embedding, cluster_number, Y, save_pred = save_pred,
                                 leiden_n_neighbors=leiden_n_neighbors, cluster_methods = cluster_methods)
    results = {**results, **res_eval}
    results["dataset"] = dataset
    results["time"] = elapsed
#     if os.path.isdir(save_to) == False:
#         os.makedirs(save_to)
#     with open(f"{save_to}/{dataset}.pickle", 'wb') as handle:
#         pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results


def cluster_embedding(embedding, cluster_number, Y, save_pred = False, 
                      leiden_n_neighbors=300, cluster_methods =["KMeans", "Leiden"]):
    """[summary]

    Args:
        embedding ([type]): [description]
        cluster_number ([type]): [description]
        Y ([type]): [description]
        save_pred (bool, optional): [description]. Defaults to False.
        leiden_n_neighbors (int, optional): [description]. Defaults to 300.

    Returns:
        [type]: [description]
    """
    result = {"t_clust" : time.time()}
    if "KMeans" in cluster_methods:
        # evaluate K-Means
        kmeans = KMeans(n_clusters=cluster_number,
                        init="k-means++",
                        random_state=0)
        pred = kmeans.fit_predict(embedding)
        if Y is not None:
            result[f"kmeans_ari"] = adjusted_rand_score(Y, pred)
            result[f"kmeans_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"kmeans_sil"] = silhouette_score(embedding, pred)
        result[f"kmeans_cal"] = calinski_harabasz_score(embedding, pred)
        result["t_k"] = time.time()
        if save_pred:
            result[f"kmeans_pred"] = pred

    if "Leiden" in cluster_methods:
        # evaluate leiden
        pred = utils.run_leiden(embedding, leiden_n_neighbors)
        if Y is not None:
            result[f"leiden_ari"] = adjusted_rand_score(Y, pred)
            result[f"leiden_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"leiden_sil"] = silhouette_score(embedding, pred)
        result[f"leiden_cal"] = calinski_harabasz_score(embedding, pred)
        result["t_l"] = time.time()
        if save_pred:
            result[f"leiden_pred"] = pred

    return result
