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
               nb_zeros=None,
               epochs = 500):
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
    
    augm_zeros=None
    random=False
    augm_value=0
    results = {}

    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs': {
            'nesterov': False,
            'weight_decay': 0.0001,
            'momentum': 0.9,
            'lr': 0.4
        },
        'scheduler': 'cosine',
        'scheduler_kwargs': {
            'lr_decay_rate': 0.1
        },
    }

    dims = [X.shape[1], 256, 64, 32, cluster_number]
    model = modelfile.STClustering(dims)
    model = model.cuda()
    for param in model.clustering.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 0.0001,)

    criterion_rep = st_loss.SupConLoss(temperature=0.07)
    criterion_rep = criterion_rep.cuda()

    batch_size = 300
    latent_repre = np.zeros((X.shape[0], dims[-1]))
    aris = []
    losses =[]
    idx = np.arange(len(X))
    perc = 0.1
    for epoch in range(epochs):
        if epoch % 10==0:
            print(".", end = "")
        model.train()
        lr = utils.adjust_learning_rate(p, optimizer, epoch)
        np.random.shuffle(idx)
        loss_=0
        for pre_index in range(len(X) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size, min(len(X), (pre_index + 1) * batch_size))
            if len(c_idx) ==0:
                continue
            c_idx = idx[c_idx]
            c_inp = X[c_idx]
            if nb_zeros == "random":
                nd1 = np.random.randint(350, 420)
                nd2 = np.random.randint(350, 420)
                input1 = torch.FloatTensor(utils.augment(c_inp, augm_zeros, nb_zeros = nd1, 
                                    perc = perc, random = random, augm_value =augm_value)).cuda()
                input2 = torch.FloatTensor(utils.augment(c_inp, augm_zeros, nb_zeros = nd2, 
                                    perc = perc, random = random, augm_value =augm_value)).cuda()
            else:
                input1 = torch.FloatTensor(utils.augment(c_inp, augm_zeros, nb_zeros = nb_zeros, 
                                        perc = perc, random = random, augm_value =augm_value)).cuda()
                input2 = torch.FloatTensor(utils.augment(c_inp, augm_zeros, nb_zeros = nb_zeros, 
                                        perc = perc, random = random, augm_value =augm_value)).cuda()
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

    model.phase = "1"
    results["features"] = features
    results1 = clustering_phase(model, X, cluster_number)
    results = {**results, **results1}
    return results


def clustering_phase(model, X, cluster_number):
    """
    Implements soft-kmeans clustering after the representation learning
    (as proposed in sczi method)
    
    Args:
        model ([type]): [description]
        X ([type]): [description]
        cluster_number ([type]): [description]

    Returns:
        [type]: [description]
    """
    alpha=0.001
    error=0.001
    gamma=0.001
    learning_rate=0.0001

    model.phase= "1"
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(X).cuda())
        features = result.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0)
    kmeans_pred = kmeans.fit_predict(features)
    last_pred = np.copy(kmeans_pred)
    
    for param in model.clustering.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    weights = torch.from_numpy(kmeans.cluster_centers_).cuda()
    model.clustering.set_weight(weights)
    dist = np.zeros((X.shape[0],cluster_number))

    model.phase= "2" # clustering phase, adds the clustering layer in the fw pass
    stop = False
    it = 0
    losses =[]

    batch_size = 300
    for epoch in range(1,200):
        model.train()
        loss_ = 0
        for pre_index in range(len(X) // batch_size + 1):
            it += 1
            if it% 140 ==0:
                Y_pred = np.argmin(dist, axis=1)
                if epoch> 10 and np.sum(Y_pred != last_pred) / len(last_pred) < error:
                    stop = True
                    break
                else:
                    last_pred = Y_pred
                

            min_idx = pre_index * batch_size
            max_idx = min(len(X), (pre_index + 1) * batch_size)
            if max_idx - min_idx ==0:
                continue
            input1 = torch.FloatTensor(X[min_idx:max_idx]).cuda()
            result = model(input1)
            total_loss = modelfile.clustering_loss(result,
                                              gamma=gamma,
                                              alpha=alpha)
            loss_ += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            dist[min_idx:max_idx] = result['latent_dist1'].detach().cpu().numpy()
        losses.append(loss_) 

        if stop:
            break

    results = {}
    results["losses"] = losses
    results["pred"] = Y_pred
    return results



def train(X,
          cluster_number,
          dataset,
          Y=None,
          n_ensemble=3,
          epochs=50,
          nb_zeros = "random",
          save_to="data/"):
    """
    Takes as input the already preprocessed X and performs representation
    learning, clustering with kmeans, leiden + soft kmeans network.
    Trains n_ensemble models to create a combined representation, also
    clustered with leiden and kmeans.
    If Y is provided, computes ari scores for all experiments.

    Args:
        X ([type]): [description]
        cluster_number ([type]): [description]
        dataset ([type]): [description]
        Y ([type], optional): [description]. Defaults to None.
        n_ensemble (int, optional): [description]. Defaults to 3.
        epochs (int, optional): [description]. Defaults to 50.
        nb_zeros (str, optional): [description]. Defaults to "random".
        save_to (str, optional): [description]. Defaults to "data/".

    Returns:
        [type]: [description]
    """
    model_results = []
    embeddings = []
    cluster_network_pred = []
    for i in range(n_ensemble):
        r = representation_phase(X, cluster_number, nb_zeros=nb_zeros, epochs=epochs)

        embeddings.append(r['features'])
        cluster_network_pred.append(r['pred'])
        model_results.append(r)
        print(f"|", end = "")
        
    # evaluation
    results = evaluate(embeddings, cluster_network_pred, cluster_number, Y)
    results["dataset"] = dataset
    results["cluster_number"] = cluster_number

    save_data = results.copy()
    save_data["model"] = model_results
    if os.path.isdir(save_to) == False:
        os.makedirs(save_to)
    with open(f"{save_to}/{dataset}.pickle", 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results

def evaluate(embeddings, cluster_network_pred, cluster_number, Y):
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
        result[f"kmeans_representation_{i}"] = adjusted_rand_score(Y, pred)

        # evaluate leiden
        pred = utils.run_leiden(embeddings[i])
        result[f"leiden_representation_{i}"] = adjusted_rand_score(Y, pred)

    # combined results
    combined_embeddings = np.hstack(embeddings)
    # evaluate K-Means
    kmeans = KMeans(n_clusters=cluster_number,
                    init="k-means++",
                    random_state=0)
    pred = kmeans.fit_predict(combined_embeddings)
    result[f"COMBINED_kmeans"] = adjusted_rand_score(Y, pred)

    # evaluate leiden
    pred = utils.run_leiden(combined_embeddings)
    result[f"COMBINED_leiden"] = adjusted_rand_score(Y, pred)

    for i in range(len(cluster_network_pred)):
        result[f"network_{i}"] = adjusted_rand_score(Y,
                                                     cluster_network_pred[i])

    return result
