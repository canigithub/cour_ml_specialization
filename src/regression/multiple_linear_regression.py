
# predicting house price (multiple variables)

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from math import log
from math import sqrt

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

data = pd.read_csv('../../data/regression/kc_house_data.csv', dtype=dtype_dict)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use build-in linear regression model
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = LinearRegression(fit_intercept=True)
example_model.fit(train_data[example_features], train_data['price'])

# print example_model.intercept_, example_model.coef_

example_predictions = example_model.predict(train_data[example_features])
# print example_predictions[0]


# model: sklearn linear regression model
def get_residual_sum_of_squares(model, data, outcome):

    predictions = model.predict(data)
    residual = outcome - predictions
    rss = (residual*residual).sum()
    return rss


rss_example_train = get_residual_sum_of_squares(example_model, train_data[example_features], train_data['price'])
# print rss_example_train

pd.options.mode.chained_assignment = None  # default='warn'
# insert new column at the back of dataframe
train_data['bedrooms_squared'] = train_data['bedrooms'] * train_data['bedrooms']
test_data['bedrooms_squared'] = test_data['bedrooms'] * test_data['bedrooms']

train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))

train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

# print test_data['bedrooms_squared'].mean()
# print test_data['bed_bath_rooms'].mean()
# print test_data['log_sqft_living'].mean()
# print test_data['lat_plus_long'].mean()

model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

model_1 = LinearRegression(fit_intercept=True)
model_1.fit(train_data[model_1_features], train_data['price'])
# print model_1.intercept_, model_1.coef_

model_2 = LinearRegression(fit_intercept=True)
model_2.fit(train_data[model_2_features], train_data['price'])
# print model_2.intercept_, model_2.coef_

model_3 = LinearRegression(fit_intercept=True)
model_3.fit(train_data[model_3_features], train_data['price'])
# print model_3.intercept_, model_3.coef_

model_1_train_rss = get_residual_sum_of_squares(model_1, train_data[model_1_features], train_data['price'])
model_2_train_rss = get_residual_sum_of_squares(model_2, train_data[model_2_features], train_data['price'])
model_3_train_rss = get_residual_sum_of_squares(model_3, train_data[model_3_features], train_data['price'])
# print model_1_train_rss, model_2_train_rss, model_3_train_rss

model_1_test_rss = get_residual_sum_of_squares(model_1, test_data[model_1_features], test_data['price'])
model_2_test_rss = get_residual_sum_of_squares(model_2, test_data[model_2_features], test_data['price'])
model_3_test_rss = get_residual_sum_of_squares(model_3, test_data[model_3_features], test_data['price'])
# print model_1_test_rss, model_2_test_rss, model_3_test_rss


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# Estimating multiple regression coefficients using gradient descent
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

# feautres/output: column names
def get_numpy_data(dataframe, features, output):
    dataframe['constant'] = 1
    features = ['constant'] + features
    feature_matrix = dataframe[features].as_matrix()
    output_array = dataframe[output].as_matrix()
    return feature_matrix, output_array

example_features, example_output = get_numpy_data(data, ['sqft_living'], 'price')
# print example_features[0, :], example_output[0]


# feature_matrix: NxD, weights: Dx1
def predict_outcome(feature_matrix, weights):
    predictions = feature_matrix.dot(weights)
    return predictions

example_weights = np.array([1., 1.])
example_predictions = predict_outcome(example_features, example_weights)
# print example_predictions[0:2]


# RSS_gradient = -2*H.T*(y-y_hat) -> Dx1 vector
# errors: y - y_hat, feature: h
# errors: Nx1, feature: NxD
def feature_derivative(errors, feature):
    return -2*feature.T.dot(errors)

example_weights = np.array([0., 0.])
example_predictions = predict_outcome(example_features, example_weights)
example_errors = example_predictions - example_output
feature_const = example_features[:, 0]
example_derivative = feature_derivative(example_errors, feature_const)
# print example_derivative, -2*example_output.sum()


# use gradient descent to find the weights
# to find the minimum, update in the direct inverse to the gradient
def regression_gradient_descent(feature_matrix, output, init_weights, step_size, tolerance):
    converged = False
    weights = np.array(init_weights)

    while not converged:
        predictions = predict_outcome(feature_matrix, weights)
        errors = output - predictions
        derivative = feature_derivative(errors, feature_matrix)
        weights -= step_size*derivative

        gradient_magnitude = sqrt((derivative*derivative).sum())
        if gradient_magnitude < tolerance:
            converged = True

    return weights


input_features = ['sqft_living']
example_feature_matrix, example_output = get_numpy_data(train_data, input_features, 'price')
init_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7
example_weights = regression_gradient_descent(example_feature_matrix, example_output, init_weights, step_size, tolerance)
# print example_weights

input_features = ['sqft_living', 'sqft_living15']
output_feature = 'price'
feature_matrix, output = get_numpy_data(train_data, input_features, output_feature)
init_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
model_weights = regression_gradient_descent(feature_matrix, output, init_weights, step_size, tolerance)

test_model_feature_matrix, test_output = get_numpy_data(test_data, input_features, output_feature)
model_predictions_test = predict_outcome(test_model_feature_matrix, model_weights)
# print model_prediction_test[0], test_output[0]


# data: feature_matrix
def get_residual_sum_of_squares(data, outcome, weights):
    predictions = predict_outcome(data, weights)
    residuals = outcome - predictions
    rss = (residuals*residuals).sum()
    return rss

model_rss = get_residual_sum_of_squares(test_model_feature_matrix, test_output, model_weights)
print model_rss