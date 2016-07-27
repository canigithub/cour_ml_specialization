
# feature selection and LASSO (interpretation)
# normalize features

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

sales = sales.sort_values(by=['sqft_living', 'price'])

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms'] * sales['bedrooms']
sales['floors_square'] = sales['floors'] * sales['floors']

all_features = ['bedrooms', 'bedrooms_square', 'bathrooms', 'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt', 'floors', 'floors_square', 'waterfront', 'view',
            'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use sklearn LASSO model
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

def get_residual_sum_of_squares(model_, data_, output_):

    predictions_ = model_.predict(data_)
    residual_ = output_ - predictions_
    rss_ = (residual_*residual_).sum()
    return rss_

model_all = Lasso(alpha=5e2, normalize=True)
model_all.fit(sales[all_features], sales['price'])
# print model_all.coef_

train_valid_data, test_data = cv.train_test_split(sales, test_size=.1, random_state=0)
train_data, valid_data = cv.train_test_split(train_valid_data, test_size=.5, random_state=0)

l1_pens = np.logspace(1,20,13)
min_rss = float('inf')
best_pen = 0

for p in l1_pens:
    model = Lasso(alpha=p, normalize=True)
    model.fit(train_data[all_features], train_data['price'])
    rss = get_residual_sum_of_squares(model, valid_data[all_features], valid_data['price'])
    if rss < min_rss:
        min_rss = rss
        best_pen = p

# print best_pen, min_rss


# limit the # of non-zero weights: narrow down the search range

def get_l1_penalty_with_max_nonzero(data, feature_names, output_name,
                                    max_nz, l1_min=1, l1_max=1e4, max_pass=3):
    count = 1
    while count < 3:
        count += 1
        l1_pens = np.logspace(log(l1_min, 10), log(l1_max, 10), 20)
        for p in l1_pens:
            model = Lasso(alpha=p, normalize=True, max_iter=1e5)
            model.fit(train_data[all_features], train_data['price'])
            nz = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
            if nz > max_nz:
                l1_min = p
            else:
                l1_max = p
                break

    return l1_max


# print get_l1_penalty_with_max_nonzero(train_data, all_features, 'price', 7)


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# Normalize features and coordinate descent for LASSO
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

sales['floors'] = sales['floors'].astype(int)

# feautres/output: column names
def get_numpy_data(dataframe, features, output):
    dataframe['constant'] = 1
    features = ['constant'] + features
    feature_matrix = dataframe[features].as_matrix()
    output_array = dataframe[output].as_matrix()
    return feature_matrix, output_array


# feature_matrix: NxD, weights: Dx1
def predict_output(feature_matrix, weights):
    predictions = feature_matrix.dot(weights)
    return predictions


# normalize columns
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)  # 2-norms of column
    return feature_matrix / norms, norms


# argmin_{w[i]} [ SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|) ]
#         / (ro[i] + lambda/2)    if ro[i] < -lambda/2
# w[i] = {  0                     if -lambda/2 <= ro[i] <= lambda/2
#         \ (ro[i] - lambda/2)    if ro[i] > lambda/2
# ro[i] = SUM[ [feature_i]*(output - (prediction -[feature_i]*w[i])) ]
# ( row 1: -H(1,i)*w[i], row 2: -H(2,i)*w[i], ... )
# note: we do not regularize constant feature, so w[0] = ro[i]

# optimizes the cost function over a single coordinate
# compute the new weight of the ith feature
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    prediction = predict_output(feature_matrix, weights)
    ro_i = feature_matrix[:, i].dot(output - prediction + feature_matrix[:, i]*weights[i])

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.

    return new_weight_i


# cyclical coordinate descent until the max step in a pass < tolerance
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights,
                                      l1_penalty, tolerance):
    weights = np.array(initial_weights)

    while True:
        max_step = 0
        for i in range(len(weights)):
            new_wi = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            if abs(new_wi - weights[i]) > max_step:
                max_step = abs(new_wi - weights[i])
            weights[i] = new_wi

        if max_step < tolerance:
            break

    return weights


feature_name = ['sqft_living', 'bedrooms']
output_name = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

(feature_matrix, output) = get_numpy_data(sales, feature_name, output_name)
(normalized_feature_matrix, norms) = normalize_features(feature_matrix)  # normalization
weights = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
pred = normalized_feature_matrix.dot(weights)
rss = ((pred-output)*(pred-output)).sum()
# print rss, weights


train_valid_data, test_data = cv.train_test_split(sales, test_size=.1, random_state=0)
train_data, valid_data = cv.train_test_split(train_valid_data, test_size=.5, random_state=0)

all_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated']
(feature_matrix, output) = get_numpy_data(sales, all_features, 'price')
(normalized_feature_matrix, norms) = normalize_features(feature_matrix)  # normalization

initial_weights = np.zeros(len(all_features)+1)
l1_penalty = 1e7
tolerance = 1
weights = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)

weights_n = weights/norms
# should print 161.31745624837794
print weights_n[3]  # the difference comes from splitting data is differnt from origianl.

