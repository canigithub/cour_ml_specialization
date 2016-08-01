# stochastic gradient ascent, variable batch size, L2 regularization
# online learning: fitting models from streaming data

from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from math import sqrt
import json

pd.options.mode.chained_assignment = None  # default='warn'

dtype_dict = {'name': str, 'review': str, 'rating': int}

products = pd.read_csv('../../data/amazon_baby.csv', dtype=dtype_dict)


def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)


# fill NaN with empty string
products = products.fillna({'review': ''})
# then remove punctuation
products['review_clean'] = products['review'].apply(remove_punctuation)
# remove rows with rating = 3
products = products[products['rating'] != 3]
# create sentiment column
products['sentiment'] = products['rating'].apply(lambda x: +1 if x > 3 else -1)

with open('../../data/important_words.json', 'r') as f:
    important_words = json.load(f)

important_words = [str(s) for s in important_words]  # convert from unicode to ascii code

# important_words = important_words[0:5]

# create a column for each important word and set value to the word count in the review
for i, word in enumerate(important_words):

    # take care of the special case 'fit' to avoid TypeError described below
    if word == 'fit':
        products['Fit'] = products['review_clean'].apply(lambda s: s.split().count(word))
        important_words[i] = 'Fit'
        continue

    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))

print 'counting important words completed.'

train_data, valid_data = train_test_split(products, test_size=.1, random_state=0)
# print 'Training set  : %d data points' % len(train_data)
# print 'Validation set: %d data points' % len(valid_data)


# features: list of column names, label: column name
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return feature_matrix, label_array


feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(valid_data, important_words, 'sentiment')


def predict_probability(feature_matrix, weights):
    scores = feature_matrix.dot(weights)
    predictions = 1 / (1 + np.exp(-scores))  # likelihood of current weights if y=1
    return predictions


# with L2 regularization
def feature_derivative_with_L2(errors, feature_matrix, weights, l2_penalty, feature_constant):
    derivatives = feature_matrix.T.dot(errors) - 2*l2_penalty*weights
    derivatives[feature_constant] += 2*l2_penalty*weights[feature_constant]
    return derivatives


'''
 log-likelihood:
 ll(w) = SUM[ I[y_i = +1]*log(P(y_i = +1 | x_i, w)) + I[y_i = -1]*log(P(y_i = -1 | x_i, w)) ]
       = SUM[ (I[y_i = +1] - 1)*score_i) - log(1 + exp(-score_i)) ]
 ll_with_L2(w) = ll(w) - l2_penalty*||w||^2
'''


# this function can be embedded in predict_probability() function.
def compute_avg_log_likelihood_with_L2(feature_matrix, sentiment, weights, l2_penalty):
    indicator = (sentiment == +1)
    scores = feature_matrix.dot(weights)
    logexp = np.log(1 + np.exp(-scores))
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    ll = ((indicator - 1) * scores - logexp).sum()
    ll_with_L2 = ll - l2_penalty * (weights * weights).sum()
    return ll_with_L2 / float(len(sentiment))


def logistic_regression_SG_with_L2(feature_matrix, sentiment, init_weights, step_size,
                                   batch_size, l2_penalty, max_iter):
    ll_avg_all = []
    weights = np.array(init_weights)
    i = 0  # index of current batch

    permut_indices = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permut_indices]
    sentiment = sentiment[permut_indices]

    for itr in range(max_iter):
        pred_proba = predict_probability(feature_matrix[i:i+batch_size], weights)
        indicator = (sentiment[i:i+batch_size] == +1)
        errors = indicator - pred_proba
        weights += step_size / float(batch_size) * feature_derivative_with_L2(
            errors, feature_matrix[i:i+batch_size], weights, l2_penalty, 0)

        ll_avg = compute_avg_log_likelihood_with_L2(feature_matrix[i:i+batch_size],
                                                    sentiment[i:i+batch_size],
                                                    weights, l2_penalty)
        ll_avg_all.append(ll_avg)

        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
                or itr % 10000 == 0 or itr == max_iter-1:
            data_size = len(feature_matrix)
            print 'Iteration %*d: Average log likelihood (of data points in batch [%0*d:%0*d]) = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr,
                 int(np.ceil(np.log10(data_size))), i,
                 int(np.ceil(np.log10(data_size))), i+batch_size, ll_avg)

        i += batch_size

        # if made a complete pass over data, shuffle and restart
        if (i + batch_size) >= len(feature_matrix):
            permut_indices = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permut_indices]
            sentiment = sentiment[permut_indices]
            i = 0

    return weights, ll_avg_all


# efficient: convolve log_likelihood_all with the vector of length
# smoothing_window that is filled with the value 1/smoothing_window.
def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9, 5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all),
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')
    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=3.0, label=label)
    plt.rcParams.update({'font.size': 16})
    # plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size': 14})


l2_pen = 1e-3
batch_size = 100
num_passes = 10
num_iterations = int(10*len(feature_matrix_train)/batch_size)

coefficients_sgd = {}
log_likelihood_sgd = {}
for step_size in np.logspace(-4, 2, num=7):
    coefficients_sgd[step_size], log_likelihood_sgd[step_size] = logistic_regression_SG_with_L2(
        feature_matrix_train, sentiment_train, np.zeros(len(important_words)+1), step_size, batch_size, l2_pen, num_iterations)


for step_size in np.logspace(-4, 2, num=7):
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=100,
              smoothing_window=30, label='step_size=%.1e' % step_size)

plt.show()