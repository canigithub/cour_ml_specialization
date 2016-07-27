
# predicting house price (one variable)

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float,
              'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float,
              'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

sales = pd.read_csv('../../data/kc_house_data.csv', dtype=dtype_dict)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)


# input_feature: one column
def simple_linear_regression(input_feature, output):

    N = float(input_feature.shape[0])
    sum_x = input_feature.sum()
    sum_y = output.sum()
    sum_xy = (input_feature*output).sum()
    sum_xx = (input_feature*input_feature).sum()
    slope = (sum_xy - sum_x*sum_y/N)/(sum_xx - sum_x*sum_x/N)
    intercept = (sum_y/N) - slope*(sum_x/N)
    return intercept, slope


test_feature = np.array(range(5))
test_output = np.array(1 + 1*test_feature)
test_intercept, test_slope =  simple_linear_regression(test_feature, test_output)
# print "Intercept: " + str(test_intercept)
# print "Slope: " + str(test_slope)

input_feature = train_data.sqft_living
output = train_data.price
sqft_intercept, sqft_slope = simple_linear_regression(input_feature, output)
# print "Intercept: " + str(sqft_intercept)
# print "Slope: " + str(sqft_slope)


# input_feature: column
def get_regression_predictions(input_feature, intercept, slope):

    predicted_values = intercept+slope*input_feature
    return predicted_values


# RSS: residual sum of square
def get_residual_sum_of_squares(input_feature, output, intercept, slope):

    predictions = get_regression_predictions(input_feature, intercept, slope)
    residuals = output - predictions
    rss = (residuals*residuals).sum()
    return rss


my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
# print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)


def inverse_regression_predictions(output, intercept, slope):
    input_feature = (output - intercept)/slope
    return input_feature


my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
# print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)

(bedrooms_intercept, bedrooms_slope) = simple_linear_regression(train_data['bedrooms'], train_data['price'])
# print "Intercept: " + str(bedrooms_intercept)
# print "Slope: " + str(bedrooms_slope)

bedrooms_RSS = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bedrooms_intercept, bedrooms_slope)
# print 'The RSS of predicting Prices based on Bedrooms is : ' + str(bedrooms_RSS)

sqft_RSS = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Sqft is : ' + str(sqft_RSS)