
# logistic regression, text cleaning, fill NaN

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import json
from math import sqrt

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

train_data, test_data = train_test_split(products, test_size=.2, random_state=0)

# print test_data[10:13]['sentiment']
# print test_data.iloc[[24937, 23773, 8562, 14154, 23747]]


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# Implement logstic regression from scratch
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

with open('../../data/important_words.json', 'r') as f:
    important_words = json.load(f)

important_words = [str(s) for s in important_words]  # convert from unicode to ascii code


# create a column for each important word and set value to the word count in the review
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))


# count how many reviews contains word perfect
products['contains_perfect'] = products['perfect'].apply(lambda x: 1 if x > 0 else 0)
# # print products['contains_perfect'].sum()


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
# estimate ranges between 0 and 1.
# feature_matrix: NxD, weights: Dx1
# return Nx1
def predict_probability(featurea_matrix, weights):
    score = featurea_matrix.dot(weights)
    probabilities = 1 / (1 + np.exp(-score))
    return probabilities


'''
 log-likelihood:
 ll(w) = SUM[ I[y_i = +1]*log(P(y_i = +1 | x_i, w)) + I[y_i = -1]*log(P(y_i = -1 | x_i, w)) ]
       = SUM[ (I[y_i = +1]-1)*score_i) - log(1 + exp(-score_i)) ]
'''


# return: a single value
def compute_log_likelihood(feature_matrix, sentiment, weights):
    indicator = (sentiment == +1)
    scores = feature_matrix.dot(weights)
    log_part = np.log(1 + np.exp(-scores))
    # check to prevent overflow
    mask = np.isinf(log_part)
    log_part[mask] = -scores[mask]
    ll = ((indicator - 1)*scores - log_part).sum()
    return ll


# Deriv over w_j = SUM[ h_1(x)*errors + ... + h_N(x)*errors ]
# errors: I[y = +1] - P(y = +1 | x, w): Nx1
# feature_matrix H: NxD,  return: Dx1
def feature_derivative(errors, feature_matrix):
    return feature_matrix.T.dot(errors)


# return the weights that maximize log-likelihood
def logistic_regression(feature_matrix, sentiment, init_weights, step_size, max_iter):
    weights = np.array(init_weights)
    indicator = (sentiment == +1)
    for itr in range(max_iter):
        prediction = predict_probability(feature_matrix, weights)
        errors = indicator - prediction
        weights += step_size * feature_derivative(errors, feature_matrix)

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or \
                (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            ll = compute_log_likelihood(feature_matrix, sentiment, weights)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, ll)

    return weights


feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
print feature_matrix.shape, sentiment.shape

weights = logistic_regression(feature_matrix, sentiment, init_weights=np.zeros(len(important_words)+1),
                              step_size=1e-7, max_iter=301)

scores = feature_matrix.dot(weights)
prediction = np.ones(len(scores))
prediction[scores <= 0] = -1
num_mistakes = (sentiment != prediction).sum()
accuracy = 1 - float(num_mistakes)/len(sentiment)
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', len(products) - num_mistakes
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.2f' % accuracy

# exit()


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use sklearn Logistic Regression model
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

'''
 '\b': backspace, r'\b' == '\\b'. r'\b\w+\b' == '\\b\\w+\\b'
 r'' stands for raw string
 returns an object to create doc-term matrix
'''
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

'''
 fit: learn a vocabulary dictionary of all tokens in raw documents
 transform: transform documents to document-term matrix:
               word1 word2 word3
       doc1      1     1     0
       doc2      1     2     0
       doc3      0     0     1
'''
train_matrix = vectorizer.fit_transform(train_data['review_clean'])

# transform the test data in the same way (use the same vocabulary)
test_matrix = vectorizer.transform(test_data['review_clean'])

sentiment_model = LogisticRegression(random_state=0)
sentiment_model.fit(train_matrix, train_data['sentiment'])

sample_test_data = test_data[10:13]
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)

# find 20 reviews in test_data with highest probability of being
# classified as positive
scores = sentiment_model.decision_function(test_matrix)
sorted_index = np.argsort(scores)
print test_data.iloc[sorted_index[-5:]]['review']


# prediction: np.ndarray, true_labels: pd.Series
# prediction == true_labels return a pd.Series
def get_classification_accuracy(model, data, true_labels):
    prediction = model.predict(data)
    return len(true_labels[prediction == true_labels]) / float(len(true_labels))

print get_classification_accuracy(sentiment_model, test_matrix, test_data['sentiment'])


# classify with significant words only
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
                     'well', 'able', 'car', 'broke', 'less', 'even', 'waste',
                     'disappointed', 'work', 'product', 'money', 'would', 'return']

vectorizer_subset = CountVectorizer(vocabulary=significant_words)  # limit to 20 words
train_matrix_subset = vectorizer_subset.fit_transform(train_data['review_clean'])
test_matrix_subset = vectorizer_subset.transform(test_data['review_clean'])

simple_model = LogisticRegression(random_state=0)
simple_model.fit(train_matrix_subset, train_data['sentiment'])
# print len(simple_model.coef_[0])  # result is 20
# print len(simple_model.coef_.flatten())

simple_model_coef_table = pd.DataFrame({'word': significant_words,
                                        'coefficient': simple_model.coef_.flatten()})
print simple_model_coef_table
