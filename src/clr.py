import numpy as np
import pandas as pd
import torch
def load_file_as_Adj_matrix(filename):
    import scipy.sparse as sp
    AlledgeTrain = pd.read_csv(filename,header=None)
    print(max(AlledgeTrain[1]))
    sum_AllNode = max(max(AlledgeTrain[0]),max(AlledgeTrain[1]))
    print(sum_AllNode)
    relation_matrix = np.zeros((sum_AllNode+1,sum_AllNode+1))
    for i, j in np.array(AlledgeTrain):
        lnc, mi = int(i), int(j)
        relation_matrix[lnc, mi] = 1
    Adj = sp.csr_matrix(relation_matrix, dtype=np.float32)
    return Adj
    
import scipy.sparse as sp
def load_data(adj,node_features):
    features = sp.csr_matrix(node_features, dtype=np.float32)  
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) 
    features = torch.FloatTensor(np.array(features.todense()))  
    adj = sparse_mx_to_torch_sparse_tensor(adj)   
    return adj, features
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  
    r_inv = np.power(rowsum, -1).flatten()  
    r_inv[np.isinf(r_inv)] = 0.  
    r_mat_inv = sp.diags(r_inv)  
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) 
    correct = preds.eq(labels).double()  
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):    
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)