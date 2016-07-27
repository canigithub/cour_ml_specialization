
# logistic regression, text cleaning, fill NaN

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from math import exp

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
# exit()


# '\b': backspace, r'\b' == '\\b'. r'\b\w+\b' == '\\b\\w+\\b'
# r'' stands for raw string
# returns an object to create doc-term matrix
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

# fit: learn a vocabulary dictionary of all tokens in raw documents
# transform: transform documents to document-term matrix:
#               word1 word2 word3
#       row1      1     1     0
#       row2      1     2     0
#       row3      0     0     1
train_matrix = vectorizer.fit_transform(train_data['review_clean'])

# transform the test data in the same way (use the same vocabulary)
test_matrix = vectorizer.transform(test_data['review_clean'])


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# Implement logstic regression from scratch
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #







# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use sklearn Logistic Regression model
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

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
