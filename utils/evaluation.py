import numpy as np
from sklearn import metrics
try:
    from munkres import Munkres
except ImportError:
    Munkres = None
    from scipy.optimize import linear_sum_assignment

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(label, pred_adjusted)
    purity = purity_score(label, pred)
    return nmi, ari, f, acc, purity

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    if Munkres is not None:
        indices = Munkres().compute(cost_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        indices = list(zip(row_ind, col_ind))
    kmeans_to_true_cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        kmeans_to_true_cluster_labels[i] = indices[i][1]
    
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred
