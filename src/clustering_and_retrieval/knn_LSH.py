
# week 2, knn, implement locality sentitive hashing
# a popular variant of LSH: random binary projection, approximates cosine distance
# pandas.concat


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np                      # dense matrix
from scipy.sparse import csr_matrix     # sparse matrix
from scipy.sparse.linalg import norm
import json
import copy     # shallow copy/deep copy (nested object)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from itertools import combinations
import time


pd.options.mode.chained_assignment = None  # default='warn'


# wiki: global variable in entire script
wiki = pd.read_csv('../../data/clustering_and_retrieval/people_wiki.csv')


with open('../../data/clustering_and_retrieval/people_wiki_map_index_to_word.json', 'r') as f:
    map_index_to_word = json.load(f)


# load pre-computed word count, tfidf
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']           # data array
    indices = loader['indices']     # row index array
    indptr = loader['indptr']       # col index array
    shape = loader['shape']         # len(row) x len(col)
    return csr_matrix((data, indices, indptr), shape)


# row: each doc, col: each word
corpus = load_sparse_csr('../../data/clustering_and_retrieval/people_wiki_tf_idf.npz')
# print corpus.get_shape()


# generate a collection of random vectors from standard Gaussian distribution
def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)


np.random.seed(0)

# now generate random vector in the same dimension as vocabulary size (547979)
# and 16 vectors lead to a 16-bit encoding of bin index of each doc
random_vectors = generate_random_vectors(num_vector=16, dim=547979)
# print random_vectors.shape

doc = corpus[0, :]  # first document
index_bits = (doc.dot(random_vectors) >= 0)


# given a numpy boolean array, return decimal value
# give a matrix, return a list of decimal value
def bin_to_dec(bin_mat, num_bit=16):
    """
    :param bin_mat: numpy binary matrix, N x num_bit
    :param num_bit: number of bits
    :return: decimal value of bin_mat in each row, N x 1 matrix
    """
    powers_of_two = (1 << np.arange(num_bit-1, -1, -1))
    return bin_mat.dot(powers_of_two)


# data: (sparse) matrix: N x D
def train_lsh(data, num_vector=16, seed=None):

    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)  # D x num_vec

    table = {}

    # partition data points into bins
    bin_hashcode = (data.dot(random_vectors) >= 0)  # N x num_vec
    # print type(bin_hashcode)    # <type 'numpy.ndarray'>

    # encode bin index bits into decimal
    hashcode = bin_to_dec(bin_hashcode, num_bit=num_vector)  # N x 1

    for data_index, hc in enumerate(hashcode):
        if hc not in table:
            table[hc] = [data_index]
        else:
            table[hc].append(data_index)

    model = {'data': data,                      # data matrix:  N x D
             'bin_index_bits': bin_hashcode,    # bin hashcode: N x num_vec
             'bin_indices': hashcode,           # dec hashcode: N x 1
             'table': table,                    # dict: hashcode -> [ doc indices ]
             'random_vectors': random_vectors,  # random vec:   D x num_vec
             'num_vector': num_vector           # number of bits of hashcode
             }

    return model


model = train_lsh(corpus, num_vector=16, seed=143)
hc_obama = corpus[35817].dot(model['random_vectors']) >= 0
# print bin_to_dec(hc_obama)
hc_biden = corpus[24478].dot(model['random_vectors']) >= 0
# print bin_to_dec(hc_biden)

# print np.array(model['bin_index_bits'][22745], dtype=int)  # list of 0/1's
# print (model['bin_index_bits'][35817] == model['bin_index_bits'][22745]).sum()

doc_ids = list(model['table'][model['bin_indices'][35817]])
doc_ids.remove(35817)

docs = wiki.iloc[doc_ids]
# print docs

# x, y: 1-d vector or 1xD matrix (sparse)
def cosine_distance(x, y):
    xy = x.dot(y.T)         # should be 1x1 mat
    dist = xy / norm(x) / norm(y)
    return 1 - dist[0, 0]   # ~ arccos()

obama_tfidf = corpus[35817]
biden_tfidf = corpus[24478]

# turns out Joe Biden should be much closer to Barack Obama than any of these four
# print '================= Cosine distance from Barack Obama'
# print 'Barack Obama - {0:24s}: {1:f}'.format('Joe Biden',
#                                              cosine_distance(obama_tfidf, biden_tfidf))
# for doc_id in doc_ids:
#     doc_tf_idf = corpus[doc_id]
#     print 'Barack Obama - {0:24s}: {1:f}'.format(wiki.iloc[doc_id]['name'],
#                                                  cosine_distance(obama_tfidf, doc_tf_idf))



def search_nearby_bins(query_bin_bits, table, max_search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.

    @:param query_bin_bits: numpy binary vector, shape: 1 x num_vectors
    :return: all candidates within search radius
    """

    candidates = copy.copy(initial_candidates)
    num_bits = query_bin_bits.shape[1]

    for r in range(max_search_radius + 1):

        # rev_bits_list to be explored
        rev_bits_list = combinations(range(num_bits), r)
        for bits in rev_bits_list:  # for each combination

            temp = np.array(query_bin_bits, dtype=bool)  # 1 x num_bits
            for bit in bits:        # inverse each bit
                temp[0][bit] = not temp[0][bit]

            # temp is one combination of reversed bits now
            hc = bin_to_dec(temp, num_bits)[0]  # hashcode of temp
            if hc in table:                     # if hc is in table
                doc_ids = table[hc]             # doc ids of the same hc
                for doc_id in doc_ids:
                    candidates.add(doc_id)

    return candidates


def query(vec, model, k, max_search_radius):
    """
    :param vec: query (sparse) vector, 1 x D
    :param model: LSH model
    :param k: number of nearest neighbor
    :param max_search_radius: max_bits to be flipped
    :return: sorted dataframe: doc_id: distance, and length of candidates
    """

    query_bin_bits = vec.dot(model['random_vectors']) >= 0
    candidates = list(search_nearby_bins(query_bin_bits, model['table'],
                                         max_search_radius=max_search_radius))

    cand_docs = model['data'][candidates]  # sparse matrix is still iterable over rows

    # cos_dists = []
    # for doc in cand_docs:
    #     cos_dists.append(pairwise_distances(vec, doc, metric='cosine'))

    # pairwise_distances: return dist of each pair between X and Y
    cos_dists = pairwise_distances(cand_docs, vec, metric='cosine').flatten()

    df = pd.DataFrame(index=candidates, data=cos_dists, columns=['distance'])
    df = df.sort_values(by='distance')

    return df.iloc[0:k], len(candidates)

nn = query(corpus[35817], model, 10, 3)[0]
wiki_nn = pd.concat([wiki, nn], axis=1, join='inner')


# ************************************************************
# explore effect of max_search_radius for a single query
# ************************************************************

display_features = ['name', 'distance']
num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in range(17):
    start = time.time()

    # Perform LSH query using Barack Obama, with max_search_radius
    result, num_candidates = query(corpus[35817], model, k=10,
                                   max_search_radius=max_search_radius)
    end = time.time()
    query_time = end-start  # Measure time

    print 'Radius:', max_search_radius
    # Display 10 nearest neighbors, along with document ID and name
    # print result.join(wiki[['id', 'name']], on='id').sort('distance')
    # print pd.concat([wiki, result], axis=1, join='inner')[display_features]

    # Collect statistics on 10 nearest neighbors
    average_distance_from_query = result['distance'].iloc[1:].mean()
    print average_distance_from_query
    max_distance_from_query = result['distance'].iloc[1:].max()
    min_distance_from_query = result['distance'].iloc[1:].min()

    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)


plt.figure(figsize=(7, 4.5))
plt.plot(num_candidates_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('# of documents searched')
plt.rcParams.update({'font.size':16})
# plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(query_time_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size':16})
# plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
# plt.tight_layout()
# plt.show()
