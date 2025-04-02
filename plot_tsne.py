from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as sio
import numpy as np
from matplotlib.colors import ListedColormap
import hdf5storage

def load_data_labels(dataset):
    data = hdf5storage.loadmat('./data/' + dataset + '.mat')
    # data = sio.loadmat('../IMC/DBONet-ins/data/' + dataset + '.mat')
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    return labels

def plot_tsne(array, gnd,save_path ):
    X_tsne = TSNE(n_components=2, learning_rate=100, random_state=0).fit_transform(array)
    color_palette = [
        "#b3e19b", "#67a8cd", "#ffc17f", "#cf9f88", "#6fb3a8", "#a2d2e7",
        "#50aa4b", "#ff9d9f", "#f36569", "#3581b7", "#cdb6da", "#704ba3",
        "#9a7fbd", "#dba9a8", "#e99b78", "#ff8831"
    ]
    custom_cmap = ListedColormap(color_palette)
    plt.figure(figsize=(12, 12))
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=gnd, cmap=custom_cmap, alpha=0.3)
    # sc=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=gnd,  cmap='Paired', alpha = 0.5)
    # sc=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=gnd,  cmap='Spectral')
    # for i in range(1, 11):
    #     idx = labels == i
    #     # color_idx = (i - 1) % len(color_palette)
    #     color_idx = i - 1
    #     plt.scatter(X_tsne[idx[0], 0], X_tsne[idx[0], 1],
    #                 color=color_palette[color_idx], marker='o', label=f"Category {i}", s=30, alpha=0.3, )
    #     # edgecolors=edge_color, linewidth=0.1, )
    #
    # plt.legend(ncol=10, fontsize=12, bbox_to_anchor=(0.5, 1), borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])
    # bwith = 2
    # plt.gca().spines['bottom'].set_linewidth(bwith)
    # plt.gca().spines['left'].set_linewidth(bwith)
    # plt.gca().spines['top'].set_linewidth(bwith)
    # plt.gca().spines['right'].set_linewidth(bwith)
    plt.axis('on')
    # plt.savefig(save_path)
    # plt.title(title)

    plt.savefig(save_path)
    plt.show()

data = "cifar10"
# labels = load_data_labels(data)
# print(labels)
# methods = {'SCMVC','UDBGL','RCAGL','MVSC-HFD','LMVSC','FMVACC','FDAGF','FastMICE','EMVGC','AWMVC','MvC-Net','CVCL'}
# methods = {'UDBGL','RCAGL','MVSC-HFD','LMVSC','FMVACC','FDAGF','FastMICE','EMVGC','AWMVC','MvC-Net'}
methods = {'CVCL'}
for method in methods:
    data_resp = sio.loadmat(f'./tsne/{method}.mat')
    resp = data_resp['rep']
    labels = data_resp['labels']
    print(labels)
    save_path = f'./tsne/pict/{method}2.svg'
    plot_tsne(resp, labels, save_path)

