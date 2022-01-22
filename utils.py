from doctest import testfile
from tokenize import group
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
import os
import time
import  networkx as nx
import  scipy.sparse as sp
from scipy.sparse import csr_matrix
import utils

plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def train(epoch,model,train_loader,optimizer,criterion,device,adj=None):
    model.train()
    train_bar = tqdm(train_loader)
    running_loss = 0.0
    for data, label in train_bar:

        label=label.to(device)
        optimizer.zero_grad()

        if adj is not None:
            data=data[0].T#[T,C] -> [C,T]
            data=csr_matrix(data).tolil()#######################################
            data=utils.preprocess_features(data)
            i = torch.from_numpy(data[0]).long().to(device)
            v = torch.from_numpy(data[1]).to(device)
            feature = torch.sparse.FloatTensor(i.t(), v, data[2]).to(device)
            outputs=model(feature,adj)
        else:
            data=data.to(device)
            outputs=model(data)

        loss = criterion(outputs, label)
        loss=loss.mean()

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    print("train epoch[{}] loss:{:.3f}".format(epoch+1,running_loss/len(train_loader)))


def test(epoch,model,test_loader,criterion,n_class,device,adj=None):
    t_start=time.perf_counter()
    model.eval()
    acc=0.0
    running_loss = 0.0
    confusion_matrix = torch.zeros(n_class, n_class)
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            
            label=label.to(device)

            if adj is not None:
                data=data[0].T#[T,C] -> [C,T]
                data=csr_matrix(data).tolil()#######################################
                data=utils.preprocess_features(data)
                i = torch.from_numpy(data[0]).long().to(device)
                v = torch.from_numpy(data[1]).to(device)
                feature = torch.sparse.FloatTensor(i.t(), v, data[2]).to(device)
                outputs=model(feature,adj)
            else:
                data=data.to(device)
                outputs=model(data)
            
            loss = criterion(outputs, label)
            loss=loss.mean()
            running_loss+=loss.item()
        
            predict_y=torch.max(outputs,dim=1)[1]
            label_y=torch.argmax(label)
            confusion_matrix[label_y.long(), predict_y.long()] += 1
            if predict_y[0] == label_y:
                acc += 1
    t_end=time.perf_counter()
    t_mean=(t_end-t_start)/len(test_loader)

    val_acc=acc/len(test_loader)
    confusion_matrix=confusion_matrix.detach().cpu().numpy()
    confusion_matrix=np.rint(100*confusion_matrix/confusion_matrix.sum(axis=1)[:, np.newaxis])

    print("test epoch[{}] loss:{:.3f}".format(epoch+1,running_loss/len(test_loader)))
    
    return confusion_matrix,val_acc,t_mean


def v_confusion_matrix(cm,class_list,title=None,save_path=None):
    cm=pd.DataFrame(cm,index=class_list,columns=class_list)

    plt.figure(figsize=(6,6))
    sns_plot=sns.heatmap(cm,annot=True,linewidth=0.5,fmt=".4g",cmap="binary",cbar=False)#cmap:'Reds/Blues','binary',YlGnBu','RdBu_r'
    sns_plot.tick_params(labelsize=10, direction='out')
    plt.xlabel("Predict Class")
    plt.ylabel("True Class")
    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path,bbox_inches="tight",dpi=300)  

def select_channel(n,items):
    N = len(items)
    set_all=[]
    for i in range(2**N):
        combo = []  
        for j in range(N):   
            if(i >> j ) % 2 == 1:  
                combo.append(items[j])
        if len(combo) == n:
            set_all.append(combo)
    return set_all

def get_adjmatrix():
    sensor_loc_list=np.array([3,11,2,12,10,9,15,16,8,5,7,1,4,6,13,14])-1#minus 1:crresponding to index
    group_list=[sensor_loc_list[:6],sensor_loc_list[6:12],sensor_loc_list[12:]]

    graph_dict={}
    for group in  group_list:
        for i in group:
            connect_node=[]
            for j in group:
                if j!=i:#no circle
                    connect_node.append(j)
            graph_dict[i]=connect_node
    graph_dict=dict(sorted(graph_dict.items(),key=lambda item:item[0]))#sort the graph_dict by key
    # print(graph_dict)

    adj=nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
    adj=preprocess_adj(adj)

    return adj

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)) # D
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_features(features):
        """
        Row-normalize feature matrix and convert to tuple representation
        """
        rowsum = np.array(features.sum(1)) # get sum of each row, [C, 1]
        r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [C]
        r_inv[np.isinf(r_inv)] = 0. # zero inf data
        r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [C, C]
        features = r_mat_inv.dot(features) # D^-1:[C, C]@X:[C, C]
        return utils.sparse_to_tuple(features) # [coordinates, data, shape], []

def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    dropout_mask=dropout_mask.bool()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


