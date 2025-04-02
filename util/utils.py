import numpy as np
import scipy.io as scio
from sklearn.neighbors import kneighbors_graph
import torch
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import h5py
import hdf5storage

### Data normalization
def normalization(data):
    maxVal = torch.max(data)
    minVal = torch.min(data)
    data = (data - minVal)//(maxVal - minVal)
    return data

### Data standardization
def standardization(data):
    rowSum = torch.sqrt(torch.sum(data**2, 1))
    repMat = rowSum.repeat((data.shape[1], 1)) + 1e-10
    data = torch.div(data, repMat.t())
    return data

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def features_to_adj(datasets,path="./data/", ):
    print("loading {} data...".format(datasets))
    try:
        matData = scio.loadmat('{}{}.mat'.format(path, datasets))
    except:
        matData = hdf5storage.loadmat('{}{}.mat'.format(path, datasets))

    if datasets in ["YTF50", "YTF100"]:
        X = matData['fea'][0]
        Y = matData['gt']
    else:
        X = matData['X'][0]
        Y = matData['Y']
    num_view = len(X)
    print("num_view = {}".format(num_view))
    labels = Y.reshape(-1, )
    num_sample = labels.shape[0]
    print("num_sample = {}".format(num_sample))
    n_cluster = len(set(np.array(labels)))
    m = n_cluster * 1  # anchor number
    Z = np.zeros((m, num_sample))  # m * n
    XX = []
    for p in range(num_view):
        scaler = StandardScaler(with_mean=True, with_std=True)
        if not isinstance(X[p], np.ndarray):
            X[p] = np.array(X[p])
        print(f"Processing element {p} of X, type: {type(X[p])}, shape: {X[p].shape}")
        X[p] = scaler.fit_transform(X[p].T)
        XX.append(X[p])

    XX = np.vstack(XX)
    U, _, _ = svds(XX.T, k=m)

    # K-means to initialize Z
    np.random.seed(12)
    kmeans = KMeans(n_clusters=m, max_iter=100, n_init=10, random_state=12)
    idx = kmeans.fit_predict(U)

    # Fill Z based on clustering results
    for i in range(num_sample):
        Z[idx[i], i] = 1

    # Initialize D and E for each view
    D = []
    E = []
    for i in range(num_view):
        C = np.dot(X[i], Z.T)
        U, _, Vt = svd(C, full_matrices=False)
        D_i = np.dot(U, Vt)
        D.append(D_i)
        E_i = X[i] - np.dot(D_i, Z)
        print("E{}".format(E_i.shape))
        print("D{}".format(D_i.shape))
        E.append(E_i)


    return X, Z, D, E, m, num_view, num_sample, labels, n_cluster