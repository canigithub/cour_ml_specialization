
# L2 regularization, logistic regression, text cleaning, fill NaN
# np.vectorize

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
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

# important_words = important_words[0:10]

# create a column for each important word and set value to the word count in the review
for i, word in enumerate(important_words):

    # take care of the special case 'fit' to avoid TypeError described below
    if word == 'fit':
        products['Fit'] = products['review_clean'].apply(lambda s: s.split().count(word))
        important_words[i] = 'Fit'
        continue

    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))

print 'counting important words completed.'


'''
 The following line raise 'TypeError: Expected sequence or array-like, got estimator'
 >>> train_data, valid_data = train_test_split(products, test_size=.2, random_state=0)
 because:
 File "/sklearn/utils/validation.py", line 112, in _num_sample 'estimator %s' % x):
 if hasattr(x, 'fit'):
    raise TypeError('Expected sequence or array-like, got estimator %s' % x)

 And the important_words[39] is 'fit' which give the products dataframe an
 attribute 'fit'. That's why TypeError is raised.

'''


# features: a list of column names, label:
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return feature_matrix, label_array


# produces probablistic estimate for P(y_i = +1 | x_i, w).
# feature_matrix: NxD, weights: Dx1
# return Nx1
def predict_probability(featurea_matrix, weights):
    score = featurea_matrix.dot(weights)
    probabilities = 1 / (1 + np.exp(-score))
    return probabilities


# feature_constant: column # of the constant-feature
def feature_derivative_with_L2(errors, feature_matrix, weights, l2_penalty, feature_constant):
    derivative = feature_matrix.T.dot(errors) - 2*l2_penalty*weights
    # for constant-feature: do not apply penalty
    derivative[feature_constant] += 2*l2_penalty*weights[feature_constant]
    return derivative


# ll = ll_without_penalty - l2_penalty*||w||^2
def compute_log_likelihood_with_L2(feature_matrix, sentiment, weights, l2_penalty):
    indicator = (sentiment == +1)
    scores = feature_matrix.dot(weights)
    log_part = np.log(1 + np.exp(-scores))
    mask = np.isinf(log_part)
    log_part[mask] = -scores[mask]
    ll = ((indicator - 1)*scores - log_part).sum() - l2_penalty*(weights*weights).sum()
    return ll


# return the weights that maximize log-likelihood
def logistic_regression_with_L2(feature_matrix, sentiment, init_weights, step_size, l2_penalty, max_iter):
    weights = np.array(init_weights)
    indicator = (sentiment == +1)
    for itr in range(max_iter):
        prediction = predict_probability(feature_matrix, weights)
        errors = indicator - prediction
        weights += step_size * feature_derivative_with_L2(errors, feature_matrix, weights, l2_penalty, 0)

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or \
                (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            ll = compute_log_likelihood_with_L2(feature_matrix, sentiment, weights, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, ll)

    return weights


# train, test spliting
train_data, valid_data = train_test_split(products, test_size=.2, random_state=0)

feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(valid_data, important_words, 'sentiment')

l2_pens = [0, 4, 10, 1e2, 1e3, 1e5]
# l2_pens = [0, 4, 10]
weights_list = []

for l2_pen in l2_pens:
    weights = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                          np.zeros(len(important_words)+1), 5e-6, l2_pen, 501)
    weights_list.append(weights)


weights_0_penalty = weights_list[0]
sorted_index = np.argsort(weights_0_penalty[1:])  # sort weights but ignore the [0] elem.
pos_words_index = sorted_index[-5:]  # now the index corresponds to important words
neg_words_index = sorted_index[0:5]

pos_words = np.array(important_words)[pos_words_index]
neg_words = np.array(important_words)[neg_words_index]

selected_words_index = np.concatenate((pos_words_index, neg_words_index))
selected_words = np.concatenate((pos_words, neg_words))

weights_list = np.array(weights_list)
selected_words_weights_table = pd.DataFrame({'word': selected_words})

# construct the word - weight table
for i, l2_pen in enumerate(l2_pens):
    selected_words_weights_table[str(l2_pen)] = weights_list[i, 1 + selected_words_index]

selected_words_weights_table = selected_words_weights_table.set_index('word')

# print selected_words_weights_table.ix['baby']
# print selected_words_weights_table


plt.rcParams['figure.figsize'] = 10, 6


# plot coefficent path with the increasing of l2_penalty
# table: important_words_weights_table
def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_pos = plt.get_cmap('Reds')
    cmap_neg = plt.get_cmap('Blues')

    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')

    for i, word in enumerate(positive_words):
        color = cmap_pos(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table.ix[word], '-', label=word, linewidth=4.0, color=color)

    for i, word in enumerate(negative_words):
        color = cmap_neg(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table.ix[word], '-', label=word, linewidth=4.0, color=color)

    plt.legend(loc='best', ncol=3, prop={'size': 16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    # plt.tight_layout()
    plt.show()

make_coefficient_plot(selected_words_weights_table, pos_words, neg_words, l2_pens)

# np.vectorize: return a function which can take an array of inputs
def get_classification_accuracy(feature_matrix, sentiment, weights):
    scores = feature_matrix.dot(weights)
    apply_threshold = np.vectorize(lambda x: 1 if x > 0 else -1)
    predictions = apply_threshold(scores)

    num_correct = (predictions == sentiment).sum()
    accuracy = float(num_correct) / len(feature_matrix)
    return accuracy


train_accuracy = {}
train_accuracy[0] = get_classification_accuracy(feature_matrix_train, sentiment_train, weights_list[0])
train_accuracy[4] = get_classification_accuracy(feature_matrix_train, sentiment_train, weights_list[1])
train_accuracy[10] = get_classification_accuracy(feature_matrix_train, sentiment_train, weights_list[2])
train_accuracy[1e2] = get_classification_accuracy(feature_matrix_train, sentiment_train, weights_list[3])
train_accuracy[1e3] = get_classification_accuracy(feature_matrix_train, sentiment_train, weights_list[4])
train_accuracy[1e5] = get_classification_accuracy(feature_matrix_train, sentiment_train, weights_list[5])

validation_accuracy = {}
validation_accuracy[0] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, weights_list[0])
validation_accuracy[4] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, weights_list[1])
validation_accuracy[10] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, weights_list[2])
validation_accuracy[1e2] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, weights_list[3])
validation_accuracy[1e3] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, weights_list[4])
validation_accuracy[1e5] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, weights_list[5])


# Build a simple report
for key in sorted(validation_accuracy.keys()):
    print "L2 penalty = %g" % key
    print "train accuracy = %s, validation_accuracy = %s" % (train_accuracy[key], validation_accuracy[key])
    print "--------------------------------------------------------------------------------"

