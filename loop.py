import argparse
import copy
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

def self_train_clustering(X,
               Y,
               cluster_number,
               args,
               augm_zeros=None,
               nb_zeros=None,
               random=False,
               perc=0.1,
               augm_value=0,
               model_name = "STClustering",
               epochs = 500):
    """
    Implements representation learning phase

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        cluster_number ([type]): [description]
        args ([type]): [description]
        augm_zeros ([type], optional): [description]. Defaults to None.
        nb_zeros ([type], optional): [description]. Defaults to None.
        random (bool, optional): [description]. Defaults to False.
        perc (float, optional): [description]. Defaults to 0.1.
        augm_value (int, optional): [description]. Defaults to 0.
        model_name (str, optional): [description]. Defaults to "STClustering".
        epochs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
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

    args.dims = [X.shape[1], 256, 64, 32, cluster_number]
    if model_name == "STClustering":
        model = modelfile.STClustering(args.dims)
    
    model = model.cuda()
    for param in model.clustering.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 0.0001,)

    criterion_rep = st_loss.SupConLoss(temperature=0.07)
    criterion_rep = criterion_rep.cuda()

    batch_size = 300
    latent_repre = np.zeros((X.shape[0], args.dims[-1]))
    aris = []
    losses =[]
    idx = np.arange(len(X))
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
        if Y is not None:
            kmeans_accuracy,kmeans_ARI, kmeans_NMI = utils.evaluate(features, Y, cluster_number ) 
            aris.append(kmeans_ARI)
        losses.append(loss_)
        
    with torch.no_grad():
        result = model(torch.FloatTensor(X).cuda())
        features = result.detach().cpu().numpy()
        pred_leiden = utils.run_leiden(features)
        ari_leiden = adjusted_rand_score(Y, pred_leiden)

    model.phase = "1"
    kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0)
    pred_kmeans_representation = kmeans.fit_predict(features)
    results["aris_kmeans_representation"] = aris
    results["ari_leiden_representation"] = ari_leiden
    results["losses_representation"] = losses
    results["pred_kmeans_representation"] = pred_kmeans_representation
    results["features"] = features
    results1 = clustering_phase(model, X, Y, cluster_number, args)
    results = {**results, **results1}
    return results

def clustering_phase(model, X, Y, cluster_number, args):
    """
    Implements soft-kmeans clustering after the representation learning
    (as proposed in sczi method)

    Args:
        model ([type]): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cluster_number ([type]): [description]
        args ([type]): [description]
    """
    model.phase= "1"
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(X).cuda())
        features = result.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0)
    kmeans_pred = kmeans.fit_predict(features)
    last_pred = np.copy(kmeans_pred)

    kmeans_ARI = np.around(adjusted_rand_score(Y, kmeans_pred), 5)
    kmeans_NMI = np.around(
        normalized_mutual_info_score(Y, kmeans_pred), 5)

    print(f"ST Phase 1 ARI {kmeans_ARI}, NMI {kmeans_NMI}")
    
    for param in model.clustering.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    weights = torch.from_numpy(kmeans.cluster_centers_).cuda()
    model.clustering.set_weight(weights)
    dist = np.zeros((X.shape[0],cluster_number))

    model.phase= "2"
    
    stop = False
    it = 0
    losses =[]
    aris = []
    batch_size = 300
    for epoch in range(1,200):
        model.train()
        loss_ = 0
        for pre_index in range(len(X) // batch_size + 1):
            it += 1
            if it% 140 ==0:
                Y_pred = np.argmin(dist, axis=1)
                accuracy = np.around(utils.cluster_acc(Y, Y_pred), 5)
                ARI = np.around(adjusted_rand_score(Y, Y_pred), 5)
                aris.append(ARI)
                NMI = np.around(normalized_mutual_info_score(Y, Y_pred), 5)
#                 print(f"accuracy {accuracy}, ARI {ARI}, NMI {NMI}")
                if epoch> 20 and np.sum(Y_pred != last_pred) / len(last_pred) < args.error:
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
                                              gamma=args.gamma,
                                              alpha=args.alpha)
            loss_ += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            dist[min_idx:max_idx] = result['latent_dist1'].detach().cpu().numpy()
        losses.append(loss_) 
        if stop:
            break
    # accuracy = np.around(cluster_acc(Y, Y_pred), 5)
    ARI = np.around(adjusted_rand_score(Y, Y_pred), 5)
    NMI = np.around(normalized_mutual_info_score(Y, Y_pred), 5)
    print(f"accuracy {accuracy}, ARI {ARI}, NMI {NMI}")
    results = {}
    results["aris"] = aris
    results["losses"] = losses
    results["pred"] = Y_pred
    return results


