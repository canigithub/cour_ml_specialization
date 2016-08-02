# week 2, sklearn nn model, choosing features for nn search

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                     # dense matrix
from scipy.sparse import csr_matrix     # sparse matrix
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


pd.options.mode.chained_assignment = None  # default='warn'


# wiki: global variable in entire script
wiki = pd.read_csv('../../data/clustering_and_retrieval/people_wiki.csv')


with open('../../data/clustering_and_retrieval/people_wiki_map_index_to_word.json', 'r') as f:
    map_index_to_word = json.load(f)


# load pre-computed word count
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']           # data array
    indices = loader['indices']     # row index array
    indptr = loader['indptr']       # col index array
    shape = loader['shape']         # len(row) x len(col)
    return csr_matrix((data, indices, indptr), shape)


# row: each doc, col: each word
word_count = load_sparse_csr('../../data/clustering_and_retrieval/people_wiki_word_count.npz')

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

# build word count matrix from scratch
# cvec = CountVectorizer(token_pattern=r'\b\w+\b')
# wiki = wiki.fillna({'text': ''})
# word_count = cvec.fit_transform(wiki.text.apply(remove_punctuation))
# print word_count.get_shape()

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

# use 'Barack Obama' as  example
# distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)  # 1st arg: word count vector
# neighbors = wiki.iloc[indices.flatten()]
# neighbors['distance'] = distances.flatten()
# neighbors = neighbors.sort_values(by=['distance'])
# print neighbors[['name', 'distance']]


# matrix: csr_matrix, map_index_to_word: dict {'word': index}
def unpack_dict(matrix, map_index_to_word):
    # convert dict to dataframe, sort then retrieve the words
    df = pd.DataFrame.from_dict(map_index_to_word, orient='index')
    df = df.sort_values(by=[0])
    vocabulary = [str(s) for s in df.index.values]

    data = matrix.data          # row i values are stored in data[indptr[i]:indptr[i+1]]
    indices = matrix.indices    # column indices are in indices[indptr[i]:indptr[i+1]]
    indptr = matrix.indptr      # if row i is empty, indptr[i] == indptr[i+1]

    num_docs = matrix.get_shape()[0]

    return [{k: v for k, v in
             zip([vocabulary[word_id] for word_id in indices[indptr[i]:indptr[i+1]]],
                 data[indptr[i]:indptr[i+1]])}
            for i in range(num_docs)]


wiki['word_count'] = unpack_dict(word_count, map_index_to_word)


# return top words according to word count
def top_words(name):
    row = wiki[wiki['name'] == name].iloc[0]  # assume name is distinct
    df = pd.DataFrame.from_dict(row['word_count'], orient='index')
    df = df.sort_values(by=[0], ascending=False)     # sort by the first column
    df.columns = ['count']
    df['word'] = df.index       # set index as a column
    df.index = range(len(df))   # set index as row number
    return df


obama_words = top_words('Barack Obama')
barrio_words = top_words('Francisco Barrio')
combined_words = obama_words.merge(barrio_words, how='inner', on='word')
combined_words = combined_words.rename(columns={'count_x': 'Obama', 'count_y': 'Barrio'})
combined_words = combined_words.sort_values(by='Obama', ascending=False)
selected_words = combined_words.iloc[0:5]['word']


# given the word count dict, check if contains all selected words
def has_words(word_dict, words):
    unique_words = set(word_dict.keys())
    words = set(words)
    return words.issubset(unique_words)

# Quiz 1, 56066
# print wiki['word_count'].apply(lambda d: has_words(word_dict=d,
#                                                    words=selected_words)).sum()

# print wiki[wiki['name'] == 'Barack Obama'].index      # 35817
# print wiki[wiki['name'] == 'George W. Bush'].index    # 28447
# print wiki[wiki['name'] == 'Joe Biden'].index         # 24478

# Quiz 2, Obama-Bush: 34.39476704, Obama-Biden: 33.07567082, Bush-Biden: 32.75667871
# dist_mat = euclidean_distances(word_count[[35817, 28447, 24478]])
# print dist_mat

# Quiz 3, ['the', 'in', 'and', 'of', 'to', 'his', 'act', 'he', 'a', 'as']
# bush_words = top_words('George W. Bush')
# combined_words = obama_words.merge(bush_words, how='inner', on='word')
# combined_words = combined_words.rename(columns={'count_x': 'Obama', 'count_y': 'Bush'})
# combined_words = combined_words.sort_values(by='Obama', ascending=False)
# selected_words = combined_words.iloc[0:10]['word']
# print list(selected_words)


tf_idf = load_sparse_csr('../../data/clustering_and_retrieval/people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)

# build tf_idf matrix from scratch
# tvec = TfidfVectorizer(token_pattern=r'\b\w+\b')
# wiki = wiki.fillna({'text': ''})
# tf_idf = tvec.fit_transform(wiki.text.apply(remove_punctuation))
# print tf_idf.get_shape()

model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)

# use 'Barack Obama' as  example again
# distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)  # 1st arg: word count vector
# neighbors = wiki.iloc[indices.flatten()]
# neighbors['distance'] = distances.flatten()
# neighbors = neighbors.sort_values(by=['distance'])
# print neighbors[['name', 'distance']]


# return top words according to tf_idf score
def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name].iloc[0]  # assume name is distinct
    df = pd.DataFrame.from_dict(row['tf_idf'], orient='index')
    df = df.sort_values(by=[0], ascending=False)     # sort by the first column
    df.columns = ['tf_idf']
    df['word'] = df.index       # set index as a column
    df.index = range(len(df))   # set index as row number
    return df

obama_tf_idf = top_words_tf_idf('Barack Obama')
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
combined_tfidf = obama_tf_idf.merge(schiliro_tf_idf, how='inner', on='word')
combined_tfidf = combined_tfidf.rename(columns={'tf_idf_x': 'Obama', 'tf_idf_y': 'Schiliro'})
combined_tfidf = combined_tfidf.sort_values(by='Obama', ascending=False)
selected_words = combined_tfidf.iloc[0:5]['word']

# Quiz 4, 14
# print wiki['tf_idf'].apply(lambda d: has_words(word_dict=d,
#                                                words=selected_words)).sum()

# Quiz 5, Obama-Biden: 123.297
# dist_mat = euclidean_distances(tf_idf[[35817, 24478]])
# print dist_mat


# Why Obama article is not similar to Biden's?
# Both wc and tfidf are propotional to word frequency.
# compute lenght of all documents
def compute_length(text):
    return len(text.split(' '))

wiki['length'] = wiki['text'].apply(compute_length)

distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)  # 1st arg: word count vector
nearest_neighbors_euclidean = wiki.iloc[indices.flatten()]
nearest_neighbors_euclidean['distance'] = distances.flatten()
nearest_neighbors_euclidean = nearest_neighbors_euclidean.sort_values(by=['distance'])
# print nearest_neighbors_euclidean[['name', 'length', 'distance']].head(10)


# doc length of Obama's 100 nearest neighbors and compare to doc length for all docs.
# plt.figure(figsize=(10.5, 4.5))
# plt.hist(wiki['length'].values, 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
#          label='Entire Wikipedia', zorder=3, alpha=0.8)
# plt.hist(nearest_neighbors_euclidean['length'].values, 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
#          label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
# plt.axvline(x=wiki[wiki['name'] == 'Barack Obama'].iloc[0]['length'], color='k', linestyle='--', linewidth=4,
#             label='Length of Barack Obama', zorder=2)
# plt.axvline(x=wiki[wiki['name'] == 'Joe Biden'].iloc[0]['length'], color='g', linestyle='--', linewidth=4,
#             label='Length of Joe Biden', zorder=1)
# plt.axis([0, 1000, 0, 0.04])
#
# plt.legend(loc='best', prop={'size': 15})
# plt.title('Distribution of document length')
# plt.xlabel('# of words')
# plt.ylabel('Percentage')
# plt.rcParams.update({'font.size': 16})
# plt.tight_layout()
# plt.show()


model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model2_tf_idf.fit(tf_idf)
distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
nearest_neighbors_cosine = wiki.iloc[indices.flatten()]
nearest_neighbors_cosine['distance'] = distances.flatten()
nearest_neighbors_cosine = nearest_neighbors_cosine.sort_values(by=['distance'])
# print nearest_neighbors_cosine[['name', 'length', 'distance']].head(10)


# compare text length distribution of euclidian metric over cosine metric
plt.figure(figsize=(10.5, 4.5))
plt.figure(figsize=(10.5, 4.5))
plt.hist(wiki['length'].values, 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'].values, 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'].values, 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'].iloc[0], color='k', linestyle='--', linewidth=4,
            label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'].iloc[0], color='g', linestyle='--', linewidth=4,
            label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size': 15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()


# drawback of cosine metric: ignores text length completely
tweet = {'act': 3.4597778278724887,
         'control': 3.721765211295327,
         'democratic': 3.1026721743330414,
         'governments': 4.167571323949673,
         'in': 0.0009654063501214492,
         'law': 2.4538226269605703,
         'popular': 2.764478952022998,
         'response': 4.261461747058352,
         'to': 0.04694493768179923}


word_indices = [map_index_to_word[word] for word in tweet.keys()]
#                               data                row id's         col id's
tweet_tf_idf = csr_matrix((tweet.values(), ([0]*len(word_indices), word_indices)),
                          shape=(1, tf_idf.shape[1]))
obama_tf_idf = tf_idf[35817]
print cosine_distances(obama_tf_idf, tweet_tf_idf)

distances, indices = model2_tf_idf.kneighbors(obama_tf_idf, n_neighbors=10)
print distances

