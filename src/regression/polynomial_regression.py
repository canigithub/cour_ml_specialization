
# validation/cross-validation

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
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


p1_data = polynomial_dataframe(sales['sqft_living'], 1)
p1_data['price'] = sales['price']

model1 = LinearRegression(fit_intercept=True)
model1.fit(p1_data['power_1'].reshape(-1, 1), p1_data['price'])

# plt.plot(p1_data['power_1'], p1_data['price'], '.',
#          p1_data['power_1'], model1.predict(p1_data['power_1'].reshape(-1, 1)), '-')
# plt.show()

p2_data = polynomial_dataframe(sales['sqft_living'], 2)
p2_data['price'] = sales['price']

model2 = LinearRegression(fit_intercept=True)
model2.fit(p2_data[['power_1', 'power_2']], p2_data['price'])

# plt.plot(p2_data['power_1'], p2_data['price'], '.',
#          p2_data['power_1'], model2.predict(p2_data[['power_1', 'power_2']]), '-')
# plt.show()

p15_data = polynomial_dataframe(sales['sqft_living'], 15)
p15_data['price'] = sales['price']

model15 = LinearRegression(fit_intercept=True)
model15.fit(p15_data[['power_1', 'power_2', 'power_3', 'power_4', 'power_5', 'power_6',
                      'power_7', 'power_8', 'power_9', 'power_10', 'power_11', 'power_12',
                      'power_13', 'power_14', 'power_15']], p15_data['price'])

# plt.plot(p15_data['power_1'], p15_data['price'], '.',
#          p15_data['power_1'], model15.predict(p15_data[['power_1', 'power_2', 'power_3',
#          'power_4', 'power_5', 'power_6', 'power_7', 'power_8', 'power_9', 'power_10',
#          'power_11', 'power_12', 'power_13', 'power_14', 'power_15']]), '-')
# plt.show()


train_validate_data, test_data = train_test_split(sales, test_size=.1, random_state=0)
train_data, validate_data = train_test_split(train_validate_data, test_size=.5, random_state=0)

poly_train = polynomial_dataframe(train_data['sqft_living'], 15)
poly_train['price'] = train_data['price']
features = poly_train.columns

validate_rss_list = []
test_rss_list = []
min_rss = float('inf')
degree = 1

poly_validate = polynomial_dataframe(validate_data['sqft_living'], 15)
poly_validate['price'] = validate_data['price']

poly_test = polynomial_dataframe(test_data['sqft_living'], 15)
poly_test['price'] = test_data['price']

def get_residual_sum_of_squares(model, data, outcome):

    predictions = model.predict(data)
    residual = outcome - predictions
    rss = (residual*residual).sum()
    return rss

for i in range(1, 16):
    model = LinearRegression(fit_intercept=True)
    model.fit(poly_train[features[0:i]], poly_train['price'])
    rss_validate = get_residual_sum_of_squares(model, poly_validate[features[0:i]], poly_validate['price'])
    validate_rss_list.append(rss_validate)
    if rss_validate < min_rss:
        min_rss = rss_validate
        degree = i

    rss_test = get_residual_sum_of_squares(model, poly_test[features[0:i]], poly_test['price'])
    test_rss_list.append(rss_test)
    print i, rss_validate, rss_test


plt.plot(np.array(range(1, 16)), test_rss_list, 'r')
plt.show()




