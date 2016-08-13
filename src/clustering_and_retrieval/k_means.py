
# week 3, kmean, smart initialization
# numpy apply: np.apply_along_axis
# np.allclose
# invert dict

import json
import time
import pandas as pd
import numpy as np                   # dense matrix
from scipy.sparse import csr_matrix  # sparse matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import sys
import os

pd.options.mode.chained_assignment = None  # default='warn'

# wiki: global variable in entire script
# columns: URI, name, text
wiki = pd.read_csv('../../data/clustering_and_retrieval/people_wiki.csv')


# load npz file as csr_matrix
def load_sparse_csr(filename):
    loader = np.load(filename)
    indptr = loader['indptr']
    data = loader['data']           # datum of row i: data[indptr[i]:indptr[i+1]]
    indices = loader['indices']     # col indices of row i: indices[indptr[i]:indptr[i+1]]
    shape = loader['shape']
    return csr_matrix((data, indices, indptr), shape)

# row: each doc, col: each word
tf_idf = load_sparse_csr('../../data/clustering_and_retrieval/people_wiki_tf_idf.npz')
tf_idf = normalize(tf_idf)

with open('../../data/clustering_and_retrieval/people_wiki_map_index_to_word.json', 'r') as f:
    map_index_to_word = json.load(f)    # word: index


def get_initial_centroids(data, k, seed=None):
    """
    Randomly choose k data points as initial centroids
    """
    if seed is not None:
        np.random.seed(seed)

    num_data = data.shape[0]
    data_indices = np.random.randint(low=0, high=num_data, size=k)
    return data[data_indices].toarray()


def assign_distances(data, centroids):
    """
    Assign distances to the centroid to data given centroids
    :param data: N x D matrix
    :param centroids: k x D matrix
    :return: clusters assignments N x 1 float numpy array
    """
    distances = pairwise_distances(data, centroids, metric='euclidean')
    dists = np.apply_along_axis(min, 1, distances)
    return dists


def smart_initialize(data, k, seed=None):
    """
    Use k-means++ to initialize a good set of centroids
    """
    if seed is not None:
        np.random.seed(seed)

    centroids = np.zeros((k, data.shape[1]))
    num_data = data.shape[0]

    for i in range(k):
        if i == 0:
            # random select a data point as centroid
            data_index = np.random.randint(0, num_data)
            centroids[i] = data[data_index].toarray()
            continue

        dists = assign_distances(data, centroids[0:i])
        dists = dists * dists  # sample proba propotional to dist^2
        dists = dists / dists.sum()
        data_index = np.random.choice(range(num_data), p=dists)
        centroids[i] = data[data_index].toarray()

    return centroids


def assign_clusters(data, centroids):
    """
    Assign clusters to data given centroids
    :param data: N x D matrix
    :param centroids: k x D matrix
    :return: clusters assignments N x 1 int array
    """
    distances = pairwise_distances(data, centroids, metric='euclidean')
    cluster = np.apply_along_axis(np.argmin, 1, distances)
    return cluster


def revise_centroids(data, k, cluster):
    """
    Find new centroids given data cluster
    :param data: N x D matrix
    :param k: number of cluster
    :param cluster: N x 1 array
    :return: new centroids: k x D numpy array
    """
    new_centroid = []
    for i in range(k):
        mass_center = data[cluster == i].mean(axis=0)
        new_centroid.append(mass_center)
    new_centroid = np.array(new_centroid).reshape(k, -1)
    return new_centroid


def compute_heterogeneity(data, k, centroids, cluster):
    """
    Compute the 'score' of a partition: sum of squared distance to centroids
    :param data: N x D matrix
    :param k: number of cluster
    :param centroids: k x D matrix
    :param cluster: N x 1 array
    :return: scalar
    """
    score = 0
    for i in range(k):
        if data[cluster == i].shape[0] > 0:
            # note: the line below generate different result from the codes followed.
            # score += pairwise_distances(data[cluster == i], [centroids[i]], metric='l2').sum()
            dists = pairwise_distances(data[cluster == i], [centroids[i]], metric='euclidean')
            dists = dists * dists
            score += dists.sum()
    return score


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    """
    This function runs k-means on given data and initial set of centroids.
    maxiter: maximum number of iterations to run.
    :param data: N x D matrix
    :param k: number of cluster
    :param initial_centroids: k x D matrix
    :param maxiter: max number of iterations
    :param record_heterogeneity: a list, to store the history of heterogeneity as function
                                 of iterations. if None, do not store the history.
    :param verbose: if True, print how many data points changed their cluster labels in
                    each iteration
    :return: final centrioids and clusters
    """
    centroids = initial_centroids[:]
    prev_cluster = None

    for itr in range(maxiter):

        cluster = assign_clusters(data, centroids)
        centroids = revise_centroids(data, k, cluster)

        if prev_cluster is not None and (prev_cluster == cluster).all():
            break

        if verbose and prev_cluster is not None:
            num_changed = (prev_cluster != cluster).sum()
            print itr, ': {0:5d} elements changed their cluster assignment.'.format(num_changed)

        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster)
            record_heterogeneity.append(score)

        prev_cluster = cluster[:]

    return centroids, cluster


def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7, 4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})


# k = 10
# heterogeneity = {}
# start = time.time()
# for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
#     initial_centroids = get_initial_centroids(tf_idf, k, seed)
#     centroids, cluster = kmeans(tf_idf, k, initial_centroids, maxiter=400,
#                                 record_heterogeneity=None, verbose=False)
#     # To save time, compute heterogeneity only once in the end
#     heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster)
#     print('seed={0:06d}, heterogeneity={1:.5f}, size of largest cluster={2:5d}'
#           .format(seed, heterogeneity[seed], max(np.bincount(cluster))))
#     sys.stdout.flush()
# end = time.time()
# print(end-start)
#
# k = 10
# heterogeneity_smart = {}
# start = time.time()
# for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
#     initial_centroids = smart_initialize(tf_idf, k, seed)
#     centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
#                                            record_heterogeneity=None, verbose=False)
#     # To save time, compute heterogeneity only once in the end
#     heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
#     print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
#     sys.stdout.flush()
# end = time.time()
# print(end-start)
#
# plt.figure(figsize=(8, 5))
# plt.boxplot([heterogeneity.values(), heterogeneity_smart.values()], vert=False)
# plt.yticks([1, 2], ['k-means', 'k-means++'])
# plt.rcParams.update({'font.size': 16})
# plt.show()

def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    """
    Run kmeans several time, and choose the best partition
    :return: best centroids and cluster
    """
    heterogeneity = {}

    best_heterogeneity = float('inf')
    final_centroids = None
    final_cluster = None

    for i in range(num_runs):
        if seed_list is not None:
            seed = seed_list[i]
        else:
            seed = int(time.time())
        np.random.seed(seed)

        initial_centroids = smart_initialize(data, k, seed)
        centroids, cluster = kmeans(data, k, initial_centroids, maxiter=500,
                                    record_heterogeneity=None, verbose=verbose)
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster)

        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()

        if heterogeneity[seed] < best_heterogeneity:
            best_heterogeneity = heterogeneity[seed]
            final_centroids = centroids
            final_cluster = cluster

    return final_centroids, final_cluster


def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7,4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
   # plt.tight_layout()


# the following code takes one hour
# start = time.time()
# centroids = {}
# cluster_assignment = {}
# heterogeneity_values = []
# k_list = [2, 10, 25, 50, 100]
# seed_list = [0, 20000, 40000, 60000, 80000, 100000, 120000]

# for k in k_list:
#    heterogeneity = []
#    centroids[k], cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, maxiter=400,
#                                                               num_runs=len(seed_list),
#                                                               seed_list=seed_list,
#                                                               verbose=True)
#    score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
#    heterogeneity_values.append(score)

#plot_k_vs_heterogeneity(k_list, heterogeneity_values)
#end = time.time()
#print(end-start)

filename = '../../data/clustering_and_retrieval/kmeans-arrays.npz'

heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        # print k
        sys.stdout.flush()
        centroids[k] = arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
        heterogeneity_values.append(score)

    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('File not found. Skipping.')

# plt.show()
# print centroids[2].shape
# exit()

# inv_map = {v: k for k, v in map.items()}
# map_index_to_word = {v: k for k, v in map_index_to_word.items()}
# print map_index_to_word[0]
# exit()

def visualize_document_clusters(wiki, tf_idf, centroids, cluster, k, map_index_to_word, display_content=True):
    """
    wiki: original dataframe
    tf_idf: data matrix, sparse matrix format
    map_index_to_word: SFrame specifying the mapping betweeen words and column indices
    display_content: if True, display 8 nearest neighbors of each centroid
    """

    # Visualize each cluster c
    for c in xrange(k):
        # Cluster heading
        print('Cluster {0:d}    '.format(c)),
        # Print top 5 words with largest TF-IDF weights in the cluster
        # print 'centroids:', type(centroids), centroids.shape, centroids.dtype
        idx = centroids[c].argsort()[::-1]
        # print 'idx:', type(idx), idx.shape, idx.dtype, idx[0:5]
        # print type(map_index_to_word), type(map_index_to_word['category']), type(centroids[c]), idx[0:5]
        for i in range(5):  # Print each word along with the TF-IDF weight
            print('{0:s}:{1:.3f}'.format(map_index_to_word[idx[i]],
                                         centroids[c, idx[i]])),
        print idx[0:5]

        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
            distances[cluster != c] = float('inf')  # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            # Wrap the text at 80-character mark.
            for i in xrange(8):
                text = ' '.join(wiki.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.
                      format(wiki.iloc[nearest_neighbors[i]]['name'],
                             distances[nearest_neighbors[i]], text[:90],
                             text[90:180] if len(text) > 90 else ''))


map_index_to_word = {v: k for k, v in map_index_to_word.items()}

# visualize_document_clusters(wiki, tf_idf, centroids[2], cluster_assignment[2], 2, map_index_to_word, False)
# exit()

k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word, False)
np.bincount(cluster_assignment[10])
print np.bincount(cluster_assignment[10])