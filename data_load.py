import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import utils
from collections import defaultdict

IMBALANCE_THRESH = 101

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target', type=int, default=4)    
    parser.add_argument('--k', type = int, default = 5)
    if hasattr(Trainer, 'add_args'):
        Trainer.add_args(parser)
    

    return parser

def load_data(path="data/cora/", dataset="cora"):#modified from code: pygcn
    """Load citation network dataset (cora only for now)"""
    #input: idx_features_labels, adj
    #idx,labels are not required to be processed in advance
    #adj: save in the form of edges. idx1 idx2 
    #output: adj, features, labels are all torch.tensor, in the dense form
    #-------------------------------------------------------

    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    set_labels = set(labels)
    classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}
    classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}

    #ipdb.set_trace()
    labels = np.array(list(map(classes_dict.get, labels)))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    utils.print_edges_num(adj.todense(), labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    #adj = torch.FloatTensor(np.array(adj.todense()))

    return adj, features, labels


def Extract_graph(edgelist, fake_node, node_num):
    
    node_list = range(node_num+1)[1:]
    node_set = set(node_list)
    adj_1 = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(edgelist.max()+1, edgelist.max()+1), dtype=np.float32)
    adj_1 = adj_1 + adj_1.T.multiply(adj_1.T > adj_1) - adj_1.multiply(adj_1.T > adj_1)
    adj_csr = adj_1.tocsr()
    for i in np.arange(node_num):
        for j in adj_csr[i].nonzero()[1]:
            node_set.add(j)

    node_set_2 = node_set
    '''
    node_set_2 = set(node_list)
    for i in node_set:
        for j in adj_csr[i].nonzero()[1]:
            node_set_2.add(j)
    '''
    node_list = np.array(list(node_set_2))
    node_list = np.sort(node_list)
    adj_new = adj_csr[node_list,:]

    node_mapping = dict(zip(node_list, range(0, len(node_list), 1)))

    edge_list = []
    for i in range(len(node_list)):
        for j in adj_new[i].nonzero()[1]:
            if j in node_list:
                edge_list.append([i, node_mapping[j]])

    edge_list = np.array(edge_list)
    #adj_coo_new = sp.coo_matrix((np.ones(len(edge_list)), (edge_list[:,0], edge_list[:,1])), shape=(len(node_list), len(node_list)), dtype=np.float32)

    label_new = np.array(list(map(lambda x: 1 if x in fake_node else 0, node_list)))
    np.savetxt('data/twitter/sub_twitter_edges', edge_list,fmt='%d')
    np.savetxt('data/twitter/sub_twitter_labels', label_new,fmt='%d')

    return

def load_data_twitter():
    adj_path = 'data/twitter/twitter.csv'
    fake_id_path = 'data/twitter/twitter_fake_ids.csv'

    adj = np.loadtxt(adj_path, delimiter=',', skiprows=1)#(total: 16011444 edges, 5384162 nodes)
    adj = adj.astype(int)
    adj = np.array(adj,dtype=int)
    fake_node = np.genfromtxt(fake_id_path, delimiter=',',skip_header=1, usecols=(0), dtype=int)#(12437)
    
    #'''#using broad walk
    if False:
        Extract_graph(adj, fake_node, node_num=1000)

    #'''

    '''generated edgelist for deepwalk for embedding
    np.savetxt('data/twitter/twitter_edges', adj,fmt='%d')
    '''

    #process adj:
    adj[adj>50000] = 0 #save top 50000 node, start from 1
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = np.array(adj.todense())
    adj = adj[1:, 1:]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tocoo()

    fake_node = np.sort(fake_node)
    fake_node = fake_node[fake_node<=50000]
    fake_id = fake_node-1

    #process label:
    labels = np.zeros((50000,),dtype=int)
    labels[fake_id] = 1


    #filtering out outliers:
    node_degree = adj.sum(axis=1)
    chosen_idx = np.arange(50000)[node_degree>=4]
    ipdb.set_trace()


    #embed need to be read sequentially, due to the size
    embed = np.genfromtxt('data/twitter/twitter.embeddings_64', max_rows=50000)
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]
    features = normalize(feature)

    adj = adj[chosen_idx,:][:,chosen_idx]     #shape:
    labels = labels[chosen_idx]     #shape:
    features = features[chosen_idx]

    

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    utils.print_edges_num(adj.todense(), labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def load_sub_data_twitter():
    adj_path = 'data/twitter/sub_twitter_edges'
    fake_id_path = 'data/twitter/sub_twitter_labels'

    adj = np.loadtxt(adj_path, delimiter=' ', dtype=int)#
    adj = np.array(adj,dtype=int)
    labels = np.genfromtxt(fake_id_path, dtype=int)#(63167)
    
    #process adj:
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #filtering out outliers:
    node_degree = np.array(adj.sum(axis=1)).reshape(-1)
    chosen_idx = np.arange(adj.shape[0])[node_degree>=4]#44982 nodes were left

    #embed need to be read sequentially, due to the size
    embed = np.genfromtxt('data/twitter/sub_node_embedding_64', skip_header=1)
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]
    features = normalize(feature)

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)

    utils.print_edges_num(adj.todense(), labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def load_data_Blog():#
    #--------------------
    #
    #--------------------
    mat = loadmat('data/BlogCatalog/blogcatalog.mat')
    adj = mat['network']
    label = mat['group']

    embed = np.loadtxt('data/BlogCatalog/blogcatalog.embeddings_64')
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]

    features = normalize(feature)
    labels = np.array(label.todense().argmax(axis=1)).squeeze()

    labels[labels>16] = labels[labels>16]-1

    print("change labels order, imbalanced classes to the end.")
    #ipdb.set_trace()
    labels = refine_label_order(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    #adj = torch.FloatTensor(np.array(adj.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(),0,-1):
        if sum(labels==i) >= IMBALANCE_THRESH and i>j:
            while sum(labels==j) >= IMBALANCE_THRESH and i>j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels
        




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_sparse(adj):#normalize a torch dense tensor for GCN, and change it into sparse.
    adj = adj + torch.eye(adj.shape[0]).to(adj)
    rowsum = torch.sum(adj,1)
    r_inv = 1/rowsum
    r_inv[torch.isinf(r_inv)] = 0.
    new_adj = torch.mul(r_inv.reshape(-1,1), adj)

    indices = torch.nonzero(new_adj).t()
    values = new_adj[indices[0], indices[1]] # modify this based on dimensionality

    return torch.sparse.FloatTensor(indices, values, new_adj.size())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def find_shown_index(adj, center_ind, steps = 2):
    seen_nodes = {}
    shown_index = []

    if isinstance(center_ind, int):
        center_ind = [center_ind]

    for center in center_ind:
        shown_index.append(center)
        if center not in seen_nodes:
            seen_nodes[center] = 1

    start_point = center_ind
    for step in range(steps):
        new_start_point = []
        candid_point = set(adj[start_point,:].reshape(-1, adj.shape[1]).nonzero()[:,1])
        for i, c_p in enumerate(candid_point):
            if c_p.item() in seen_nodes:
                pass
            else:
                seen_nodes[c_p.item()] = 1
                shown_index.append(c_p.item())
                new_start_point.append(c_p)
        start_point = new_start_point

    return shown_index

