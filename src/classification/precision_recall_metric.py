
# logistic regression, vectorier

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

# fit: learn column names, transform: put numbers in each row
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])

model = LogisticRegression(random_state=0)
model.fit(train_matrix, train_data['sentiment'])

# print type(model.predict_proba(test_matrix))
# exit()

accuracy = accuracy_score(y_true=test_data['sentiment'].as_matrix(),
                          y_pred=model.predict(test_matrix))
print "Test Accuracy: %s" % accuracy

baseline = float(len(test_data[test_data['sentiment'] == 1]))/len(test_data)
print "Baseline accuracy (majority class classifier): %s" % baseline

cmat = confusion_matrix(y_true=test_data['sentiment'].as_matrix(),
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)

print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
# print the confusion matrix.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i, j])


precision = precision_score(y_true=test_data['sentiment'].as_matrix(),
                            y_pred=model.predict(test_matrix))
print "Precision on test data: %s" % precision

recall = recall_score(y_true=test_data['sentiment'].as_matrix(),
                      y_pred=model.predict(test_matrix))
print "Recall on test data: %s" % recall


def plot_precision_recall_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis='x', nbins=5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color='#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})


threshold_values = np.linspace(0.5, 0.9, num=100)
# print 'threshold values:', threshold_values

precision_all = []
recall_all = []

for tv in threshold_values:

    pred_proba = pd.Series(data=model.predict_proba(test_matrix)[:, 1])
    precision = precision_score(y_true=test_data['sentiment'].as_matrix(),
                                y_pred=pred_proba.apply(lambda x: +1 if x > tv else -1))
    recall = recall_score(y_true=test_data['sentiment'].as_matrix(),
                          y_pred=pred_proba.apply(lambda x: +1 if x > tv else -1))
    precision_all.append(precision)
    recall_all.append(recall)


# plt.plot(range(len(precision_all)), precision_all, '-b')
# plt.plot(range(len(recall_all)), recall_all, '-r')
plot_precision_recall_curve(precision_all, recall_all, 'Precision recall curve (all)')
plt.show()