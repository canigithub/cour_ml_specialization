
# knn, cross validation, vectorization

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from math import log, sqrt

pd.options.mode.chained_assignment = None  # default='warn'

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': float, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}


sales = pd.read_csv('../../data/kc_house_data.csv', dtype=dtype_dict)


# feautres/output: column names
def get_numpy_data(dataframe, features, output):
    dataframe['constant'] = 1
    features = ['constant'] + features
    feature_matrix = dataframe[features].as_matrix()
    output_array = dataframe[output].as_matrix()
    return feature_matrix, output_array


# in computing distances, it's crucial to normalize features
# normalize columns
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)  # 2-norms of column
    return feature_matrix / norms, norms


train_valid_data, test_data = cv.train_test_split(sales, test_size=.2, random_state=0)
train_data, valid_data = cv.train_test_split(train_valid_data, test_size=.2, random_state=0)

feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated', 'lat',
                'long', 'sqft_living15', 'sqft_lot15']
features_train, output_train = get_numpy_data(train_data, feature_list, 'price')
features_test, output_test = get_numpy_data(test_data, feature_list, 'price')
features_valid, output_valid = get_numpy_data(valid_data, feature_list, 'price')

# print features_test[0], features_train[9].shape
# print features_train[0:3] - features_test[0]


# features: matrix, query: vector.
def compute_distances(features_instances, features_query):
    diff = features_instances - features_query
    distances = np.sqrt(diff * diff).sum(axis=1)
    return distances


# feature_train: NxD, query: Dx1
# return index of the k nearest neighbor
def k_nearest_neighbors(k, feature_train, features_query):
    distances = compute_distances(feature_train, features_query)
    sorted_index = np.argsort(distances)
    return sorted_index[0:k]


# single query
def predict_output_of_query(k, features_train, output_train, features_query):
    neighbors = k_nearest_neighbors(k, features_train, features_query)
    prediction = output_train[neighbors].mean()
    return prediction


# a set of query
def predict_output(k, features_train, output_train, features_query):
    predictions = []
    for i in range(len(features_query)):
        prediction = predict_output_of_query(k, features_train,
                                             output_train, features_query[i])
        predictions.append(prediction)
    return predictions


# use cross validation to find best k
best_k = 0
min_rss = float('inf')
for k in range(1, 16):
    print k
    pred = predict_output(k, features_train, output_train, features_valid)
    residual = output_valid - pred
    rss = (residual * residual).sum()
    if rss < min_rss:
        min_rss = rss
        best_k = k

print best_k