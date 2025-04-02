'''
   This program is to evaluate clustering performance
'''

from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster._supervised import check_clusterings
import warnings
from scipy import sparse as sp
import numpy.linalg as LA
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")


def KMeansClustering(features, gnd, clusterNum, randNum):
    """
    :param features: 1d array containing the ground truth cluster labels.
    :param gnd: true labels, numpy.array with shape `(n_samples,)`
    :param clusterNum:  number of clusters
    :param randNum: random seeds
    :return: the preditive clustering label
    """
    kmeans = KMeans(n_clusters=clusterNum, n_init=1, max_iter=500,
                    random_state=randNum)
    if(np.isnan(features).any()):
        nan=np.isnan(features)
        features[nan]=0.000001
    estimator = kmeans.fit(features)
    clusters = estimator.labels_
    # print("The type of clusters is: ", type(clusters))
    # print("Clustering results are: ", clusters.shape)
    labels = np.zeros_like(clusters)
    for i in range(clusterNum):
        mask = (clusters == i)
        labels[mask] = mode(gnd[mask])[0]

    return labels


def similarity_function(points):
    """
    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind=ind.T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering_purity(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    :return: Purity
    """
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins = range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct/len(y_pred)

    return Purity

def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score, precision, recall

### Evaluation metrics of clustering performance
def clusteringMetrics(trueLabel, predictiveLabel):
    # Clustering accuracy

    ACC=metrics.accuracy_score(trueLabel, predictiveLabel)
    # Normalized mutual information
    #NMI = metrics.v_measure_score(trueLabel, predictiveLabel)
    NMI = normalized_mutual_info_score(trueLabel, predictiveLabel)
    # Purity
    Purity = clustering_purity(trueLabel.reshape((-1, 1)), predictiveLabel.reshape(-1, 1))
    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, predictiveLabel)
    Fscore, Precision, Recall = b3_precision_recall_fscore(trueLabel, predictiveLabel)

    return ACC, NMI, Purity, ARI, Fscore, Precision, Recall


### Report mean and std of 10 experiments
def StatisticClustering(features, gnd, clusterNum):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 10
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    PurityList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    FscoreList = np.zeros((repNum, 1))
    PrecisionList = np.zeros((repNum, 1))
    RecallList = np.zeros((repNum, 1))

    #clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    for i in range(repNum):
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, i)
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, predictiveLabel)

        ACCList[i] = ACC
        NMIList[i] = NMI
        PurityList[i] = Purity
        ARIList[i] = ARI
        FscoreList[i] = Fscore
        PrecisionList[i] = Precision
        RecallList[i] = Recall
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    Puritymean_std = np.around([np.mean(PurityList), np.std(PurityList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    Fscoremean_std = np.around([np.mean(FscoreList), np.std(FscoreList)], decimals=4)
    Precisionmean_std = np.around([np.mean(PrecisionList), np.std(PrecisionList)], decimals=4)
    Recallmean_std = np.around([np.mean(RecallList), np.std(RecallList)], decimals=4)
    #plt.scatter(features[:, 0], features[:, 2], c = predictiveLabel)
    #plt.savefig("Clustering_results.jpg")
    #plt.show()
    return ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std

