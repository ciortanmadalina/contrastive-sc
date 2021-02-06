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
import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import model as modelfile
import st_loss
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


def representation_phase(X,
               cluster_number,
               dropout=0.8,
               lr = 1e-5,
               epochs = 20,
               temperature = 0.07):
    """
    Performs representation learning for epochs epochs.
    nb_zeros controls the number of genes with 0  value (data augmentation)

    Args:
        X ([type]): [description]
        cluster_number ([type]): [description]
        nb_zeros ([type], optional): [description]. Defaults to None.
        epochs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
    results = {}

#     p = {
# #         'epochs': 500,
#         'optimizer': 'sgd',
#         'optimizer_kwargs': {
#             'nesterov': False,
#             'weight_decay': 0.0001,
#             'momentum': 0.9,
#             'lr': 0.4
#         },
#         'scheduler': 'cosine',
#         'scheduler_kwargs': {
#             'lr_decay_rate': 0.1
#         },
#     }

    dims = [X.shape[1], 256, 64, 32, cluster_number]
    model = modelfile.ContrastiveRepresentation(dims, dropout = dropout)
    model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= lr)

    criterion_rep = st_loss.SupConLoss(temperature=temperature)
    criterion_rep = criterion_rep.cuda()

    batch_size = 300
    latent_repre = np.zeros((X.shape[0], dims[-1]))
    aris = []
    losses =[]
    idx = np.arange(len(X))
    for epoch in range(epochs):
        if epoch % 10==0:
            print(".", end = "")
        model.train()
#         lr = utils.adjust_learning_rate(p, optimizer, epoch)
        np.random.shuffle(idx)
        loss_=0
        for pre_index in range(len(X) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size, min(len(X), (pre_index + 1) * batch_size))
            if len(c_idx) ==0:
                continue
            c_idx = idx[c_idx]
            c_inp = X[c_idx]
            input1 = torch.FloatTensor(c_inp).cuda()
            input2 = torch.FloatTensor(c_inp).cuda()
            anchors_output = model(input1)
            neighbors_output = model(input2)

            features = torch.cat([anchors_output.unsqueeze(1), neighbors_output.unsqueeze(1)], dim=1)
            total_loss = criterion_rep(features)
            loss_ +=total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            result = model(torch.FloatTensor(X).cuda())
            features = result.detach().cpu().numpy()

        losses.append(loss_)
        
    with torch.no_grad():
        result = model(torch.FloatTensor(X).cuda())
        features = result.detach().cpu().numpy()
    results["features"] = features
    return results


def train(X,
        cluster_number,
        dataset,
        Y=None,
        dropout=0.8,
        lr = 1e-5,
        epochs = 20,
        temperature = 0.07,
        save_to="data/"):
    """
    Takes as input the already preprocessed X and performs representation
    learning, clustering with kmeans, leiden + soft kmeans network.
    Trains n_ensemble models to create a combined representation, also
    clustered with leiden and kmeans.
    If Y is provided, computes ari scores for all experiments.


    Returns:
        [type]: [description]
    """
    start = time.time()

    r = representation_phase(X, cluster_number, temperature=temperature, epochs=epochs,
                            lr =lr, dropout = dropout)
    elapsed = time.time()-start
    # evaluation
    results = evaluate(r['features'], cluster_number, Y)
    results["dataset"] = dataset
    results["cluster_number"] = cluster_number
    results["time"] = elapsed
    
    save_data = results.copy()
    save_data["model"] = r
    if os.path.isdir(save_to) == False:
        os.makedirs(save_to)
    with open(f"{save_to}/{dataset}.pickle", 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results

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
    # evaluate K-Means
    kmeans = KMeans(n_clusters=cluster_number,
                    init="k-means++",
                    random_state=0)
    pred = kmeans.fit_predict(embeddings)
    result[f"kmeans_ari"] = adjusted_rand_score(Y, pred)
    result[f"kmeans_nmi"] = normalized_mutual_info_score(Y, pred)
    result[f"kmeans_pred"] = pred

    # evaluate leiden
    pred = utils.run_leiden(embeddings)
    result[f"leiden_ari"] = adjusted_rand_score(Y, pred)
    result[f"leiden_nmi"] = normalized_mutual_info_score(Y, pred)
    result[f"leiden_pred"] = pred

    return result
