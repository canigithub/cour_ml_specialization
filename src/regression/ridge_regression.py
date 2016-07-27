
# regression with L2 penalty, cross validation

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}


sales = pd.read_csv('../../data/regression/kc_house_data.csv', dtype=dtype_dict)

sales = sales.sort_values(by=['sqft_living', 'price'])


# create polynomial dataframe. asumme degree >= 1
# feature is pandas.Series
# return a matrix: 1st column is feature, last column is feature to the max-power
def polynomial_dataframe(feature, degree):
    poly_df = pd.DataFrame()
    poly_df['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            col_name = 'power_' + str(power)
            poly_df[col_name] = feature.apply(lambda x: x**power)

    return poly_df


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use build-in ridge regression model
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

l2_penalty = 1e-5

p15_data = polynomial_dataframe(sales['sqft_living'], 15)
model = Ridge(alpha=l2_penalty, fit_intercept=True, normalize=True)
model.fit(p15_data, sales['price'])
# print model.intercept_, model.coef_

train_valid_data, test_data = cv.train_test_split(sales, test_size=.1, random_state=0)
train_valid_shuffled = shuffle(train_valid_data, random_state=0)
train_data, valid_data = cv.train_test_split(train_valid_data, test_size=.5, random_state=0)


# model: sklearn linear regression model
# data: matrix, outcome: vector
def get_residual_sum_of_squares(model_, data_, outcome_):

    predictions_ = model_.predict(data_)
    residual_ = outcome_ - predictions_
    rss_ = (residual_*residual_).sum()
    return rss_


# use cross validation to find the error
def k_fold_cross_validation(k, l2_penalty, data, output):
    n = len(data)
    err = 0
    for i in range(k):
        start = n * i / k
        end = n * (i+1) / k
        valid_data_ = data[start:end]
        train_data_ = data[0:start].append(data[end:n])
        valid_output_ = output[start:end]
        train_output_ = output[0:start].append(output[end:n])
        model_ = Ridge(alpha=l2_penalty, fit_intercept=True, normalize=True)
        model_.fit(train_data_, train_output_)
        err += get_residual_sum_of_squares(model_, valid_data_, valid_output_)

    err /= k   # return the average validation error
    return err


k = 10
l2_penalties = np.logspace(1, 7, 13)
p15_train = polynomial_dataframe(train_data['sqft_living'], 15)
p15_output = train_data['price']
min_err = float('inf')
best_penalty = 0

for p in l2_penalties:
    error = k_fold_cross_validation(k, p, p15_train, p15_output)
    # print error
    if error < min_err:
        min_err = error
        best_penalty = p

# print best_penalty


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# Estimating ridge regression coefficients using gradient descent
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

# feautres/output: column names
def get_numpy_data(dataframe, features, output):
    dataframe['constant'] = 1
    features = ['constant'] + features
    feature_matrix = dataframe[features].as_matrix()
    output_array = dataframe[output].as_matrix()
    return feature_matrix, output_array







