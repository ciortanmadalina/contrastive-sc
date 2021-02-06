"""
Original implementation of Contrastive-sc method
(https://github.com/ciortanmadalina/contrastive-sc)
By Madalina Ciortan (01/10/2020)
"""
import argparse
import copy
import os
import pickle
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import models
import st_loss
import utils
import math
import train 
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

def adjust_learning_rate( optimizer, epoch):
    p = {
      'epochs': 500,
     'optimizer': 'sgd',
     'optimizer_kwargs': {'nesterov': False,
      'weight_decay': 0.0001,
      'momentum': 0.9,
      'lr': 0.4},
     'scheduler': 'cosine',
     'scheduler_kwargs': {'lr_decay_rate': 0.1},
     }
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
def evaluate(embeddings, cluster_number, Y):
    """
    Computes the ARI scores of all experiments ran as part of train method.

    Args:
        embeddings ([type]): [description]
        cluster_network_pred ([type]): [description]
        cluster_number ([type]): [description]
        Y ([type]): [description]

    Returns:
        [type]: [description]
    """
    result = {}
    if Y is None:
        return result
    for i in range(len(embeddings)):
        # evaluate K-Means
        kmeans = KMeans(n_clusters=cluster_number,
                        init="k-means++",
                        random_state=0)
        pred = kmeans.fit_predict(embeddings[i])
        result[f"kmeans_ari_{i}"] = adjusted_rand_score(Y, pred)
        result[f"kmeans_nmi_{i}"] = normalized_mutual_info_score(Y, pred)
        result[f"kmeans_pred_{i}"] = pred

        # evaluate leiden
        pred = utils.run_leiden(embeddings[i])
        result[f"leiden_ari_{i}"] = adjusted_rand_score(Y, pred)
        result[f"leiden_nmi_{i}"] = normalized_mutual_info_score(Y, pred)
        result[f"leiden_pred_{i}"] = pred

    if len(embeddings)>1:
        # combined results
        combined_embeddings = np.hstack(embeddings)
        # evaluate K-Means
        kmeans = KMeans(n_clusters=cluster_number,
                        init="k-means++",
                        random_state=0)
        pred = kmeans.fit_predict(combined_embeddings)
        result[f"COMBINED_kmeans_ari"] = adjusted_rand_score(Y, pred)
        result[f"COMBINED_kmeans_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"COMBINED_kmeans_pred"] = pred

        # evaluate leiden
        pred = utils.run_leiden(combined_embeddings)
        result[f"COMBINED_leiden_ari"] = adjusted_rand_score(Y, pred)
        result[f"COMBINED_leiden_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"COMBINED_leiden_pred"] = pred


    return result


def get_device():
    """[summary]

    Returns:
        [type]: [description]
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def train_model(X,
                cluster_number,
                Y=None,
                nb_epochs=20,
                lr=1e-5,
                temperature=0.07,
                dropout=0.8,
                evaluate = False,
                layers = [256, 64, 32]):
    device = get_device()
    dims = np.concatenate([[X.shape[1]], layers])#[X.shape[1], 256, 64, 32]
    model = models.ContrastiveRepresentation(dims, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=nb_epochs,
                                                           eta_min=0.0)

    criterion_rep = st_loss.SupConLoss(temperature=temperature)
    criterion_rep = criterion_rep.cuda()

    batch_size = 200

    losses = []
    idx = np.arange(len(X))
    for epoch in range(nb_epochs):

        model.train()
        lr = adjust_learning_rate(optimizer, epoch)
        np.random.shuffle(idx)
        loss_ = 0
        for pre_index in range(len(X) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(X), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = X[c_idx]

            input1 = torch.FloatTensor(c_inp).to(device)
            input2 = torch.FloatTensor(c_inp).to(device)

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
        if evaluate:
            model.eval()
            with torch.no_grad():
                result = model(torch.FloatTensor(X))
                features = result.detach().cpu().numpy()
            res = train.evaluate([features], cluster_number, Y)
            print(
                f"{epoch}). Loss {loss_}, ARI {res['kmeans_ari_0']}, {res['leiden_ari_0']}"
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
        nb_epochs=20,
        lr=1e-5,
        temperature=0.07,
        dropout=0.8,
        evaluate=True,
        n_ensemble=1,
        layers = [256, 64, 32],
        save_to="data/"):
    results = {}
    embeddings = []
    for i in range(n_ensemble):
        f = train.train_model(X,
                  cluster_number,
                  Y=Y,
                  nb_epochs=nb_epochs,
                  lr=lr,
                  temperature=temperature,
                  dropout=dropout,
                  layers = layers,
                  evaluate=False)
        results[f"features_{i}"] = f
        embeddings.append(f)
    if Y is not None:
        res_eval = train.evaluate(embeddings, cluster_number, Y)
    results = {**results, **res_eval}
    results["dataset"] = dataset
    if os.path.isdir(save_to) == False:
        os.makedirs(save_to)
    with open(f"{save_to}/{dataset}.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results