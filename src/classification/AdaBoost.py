# train a decision tree, one-hot encoding
# use graphviz to visulize decision tree
# tree pruning

import numpy as np
import pandas as pd
import json
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from math import log, exp
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None  # default='warn'

loans = pd.read_csv('../../data/lending_club_data.csv', low_memory=False)

loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']

# less features for implementing adaboost
target = 'safe_loans'
features = [
            'grade',                     # grade of the loan (categorical)
            # 'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            # 'short_emp',                 # one year or less of employment
            # 'emp_length_num',            # number of years of employment (for part 1)
            'emp_length',                # number of years of employment (for part 2)
            'home_ownership',            # home_ownership status: own, mortgage or rent
            # 'dti',                       # debt to income ratio
            # 'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            # 'payment_inc_ratio',         # ratio of the monthly payment to income
            # 'delinq_2yrs',               # number of delinquincies
            # 'delinq_2yrs_zero',          # no delinquincies in last 2 years
            # 'inq_last_6mths',            # number of creditor inquiries in last 6 months
            # 'last_delinq_none',          # has borrower had a delinquincy
            # 'last_major_derog_none',     # has borrower had 90 day or worse rating
            # 'open_acc',                  # number of open credit accounts
            # 'pub_rec',                   # number of derogatory public records
            # 'pub_rec_zero',              # no derogatory public records
            # 'revol_util',                # percent of available credit being used
            # 'total_rec_late_fee',        # total late fees received to day
            # 'int_rate',                  # interest rate of the loan
            # 'total_rec_int',             # interest received to date
            # 'annual_inc',                # annual income of borrower
            # 'funded_amnt',               # amount committed to the loan
            # 'funded_amnt_inv',           # amount committed by investors for the loan
            # 'installment',               # monthly payment owed by the borrower
           ]

# Extract the feature columns and target column

loans = loans[features + [target]].dropna()


# split data to train and validation
with open('../../data/module_8_assignment_1_train_idx.json') as f:
    train_idx = json.load(f)

with open('../../data/module_8_assignment_1_validation_idx.json') as f:
    valid_idx = json.load(f)

train_data = loans.iloc[train_idx]
valid_data = loans.iloc[valid_idx]

safe_loans_raw = loans[loans['safe_loans'] == +1]
risk_loans_raw = loans[loans['safe_loans'] == -1]

percentage = len(risk_loans_raw)/float(len(safe_loans_raw))
risk_loans = risk_loans_raw
safe_loans = safe_loans_raw.sample(frac=percentage, random_state=1)
# print "Number of safe loans  : %s" % len(safe_loans)
# print "Number of risky loans : %s" % len(risk_loans)

# now the load_data is balanced
loans_data = risk_loans.append(safe_loans)


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# One-hot encoder
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #


# one-hot encoder: for numerical values, divide values in 10 sections.
# for string values, represent each value individually.
# dataframe: pd.Dataframe, features: LIST of column names
def encode_one_hot(dataframe, features):

    df = pd.DataFrame(dataframe[features])  # make a copy of original dataframe

    for feat in features:
        if df[feat].dtypes == int or df[feat].dtypes == float:  # all numerical -> binary categories
            categorical = np.histogram(df[feat], bins=10)[1]
            for i in range(1, len(categorical)):
                val0 = categorical[i-1]
                val1 = categorical[i]
                if i < len(categorical)-1:
                    name = str(val0) + ' <= ' + feat + ' < ' + str(val1)
                    df[name] = df[feat].apply(lambda x: 1 if val0 <= x < val1 else 0)
                else:  # the last elem
                    name = str(val0) + ' <= ' + feat + ' <= ' + str(val1)
                    df[name] = df[feat].apply(lambda x: 1 if val0 <= x <= val1 else 0)

            del df[feat]

    dvec = DictVectorizer(sort=False)

    # orient='record' creates a list of dicts where each dict represents each row
    # the default to_dict() will do nothing to float value type
    one_hot = pd.DataFrame(dvec.fit_transform(df.to_dict(orient='record')).toarray())
    one_hot.columns = dvec.get_feature_names()
    # print dvec.get_feature_names()
    one_hot.index = df.index
    return one_hot

onehot_data = encode_one_hot(loans_data, features)
onehot_features = onehot_data.columns
onehot_data[target] = loans_data[target]

feature_list = onehot_features
if type(feature_list) == pd.indexes.base.Index:
    feature_list = feature_list.tolist()  # cast to list type to support remove()

train_data, test_data = train_test_split(onehot_data, test_size=.2, random_state=0)
# print train_data.isnull().sum().sum()


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use sklearn Decision Tree Classifier
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

grad_boost_model = GradientBoostingClassifier(random_state=0)
grad_boost_model.fit(train_data[onehot_features], train_data[target])
ada_boost_model = AdaBoostClassifier(random_state=0)
ada_boost_model.fit(train_data[onehot_features], train_data[target])


def get_classification_accuracy(model, data, output):
    predictions = model.predict(data)
    num_correct = (predictions == output).sum()
    return float(num_correct) / len(output)

# print get_classification_accuracy(grad_boost_model, test_data[onehot_features], test_data[target])
# print get_classification_accuracy(ada_boost_model, test_data[onehot_features], test_data[target])


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# Implement Adaboost ensembling
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #


'''
    target: y_1, y_2, ..., y_N
    predictions: y_hat_1, ... ,y_hat_N
    data point weights: a_1, ..., a_N

    weight of mistakes: WM(a, y) = SUM[ a_i * I[y_i != y_hat_i] ]
    weighted_error: E(a, y) = WM(a, y) / SUM[ a_i ]

    compute coefficient for stump t: w_hat_t = .5 * log( (1 - E(a, y)) / E(a, y) )

    recompute datapoints weights a_j after computing coefficient for stump t:
    a_j = a_j * exp(-w_hat_t) if f_t(x_j) == y_j else a_j * exp(w_hat_t)

    normalize datapoints weights after the recomputing phase:
    a_j = a_j / SUM[ a_j ]

'''


def get_indices_from_mask(mask):
    indices = []
    for i, m in enumerate(mask):
        if m:
            indices.append(i)
    return np.array(indices, dtype=int)


# return tuple: (lower of weights of mistakes, label)
def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    if len(labels_in_node) == 0:
        return 0, 0

    # predict all to be +1, count total weight of mistakes
    # print type(labels_in_node != +1), labels_in_node.name, len(labels_in_node), len(data_weights)
    wm_pos = data_weights[get_indices_from_mask(labels_in_node != +1)].sum()
    # predict all to be -1, count total weight of mistakes
    wm_neg = data_weights[get_indices_from_mask(labels_in_node != -1)].sum()

    return (wm_pos, +1) if wm_pos <= wm_neg else (wm_neg, -1)


# If the data is identical in each feature, this function should return None
# -> if all weighted error is same on each feature, return None.
# yes go left, now go right
def best_splitting_feature(data, features, target, data_weights):

    best_feature = None
    best_error = float('+inf')
    identical = True
    error_value = None

    for feature in features:
        yes_indices = get_indices_from_mask(data[feature] == 1)
        no_indices = get_indices_from_mask(data[feature] == 0)

        # label_in_yes = data.iloc[yes_indices][target]
        # label_in_no = data.iloc[no_indices][target]
        # keep in mind: selection mask can be used directly in dataframes, but not with numpy
        yes_part = data[data[feature] == 1]
        no_part = data[data[feature] == 0]

        wm_yes, label_yes = intermediate_node_weighted_mistakes(yes_part[target], data_weights[yes_indices])
        wm_no, label_no = intermediate_node_weighted_mistakes(no_part[target], data_weights[no_indices])

        if error_value is None:
            error_value = wm_yes + wm_no
        elif error_value != (wm_yes + wm_no):
            identical = False

        if (wm_yes + wm_no) <= best_error:  #####
            best_feature = feature
            best_error = wm_yes + wm_no

    return None if identical else best_feature


# example_data_weights = np.array(len(train_data) * [1.5])
# print best_splitting_feature(train_data, feature_list, target, example_data_weights)


class Node(object):

    def __init__(self, is_leaf, prediction, left=None, right=None, splitting_feature=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.left = left
        self.right = right
        self.splitting_feature = splitting_feature

    def __str__(self):
        return str(self.splitting_feature)


def create_leaf(target_values, data_weights):
    wm, label = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf = Node(True, label)
    return leaf


# features is LIST
def weighted_decision_tree_create(data, features, target, data_weights,
                                  current_depth=0, max_depth=10):
    remaining_features = features[:]
    target_values = data[target]
    total_weights = data_weights.sum()

    # stop conditon 1: weighted error is 0
    if intermediate_node_weighted_mistakes(data[target], data_weights)[0] \
            / float(total_weights) <= 1e-15:
        # print 'stop conditon 1 reached.'
        return create_leaf(target_values, data_weights)

    # stop conditon 2: no more features:
    if len(remaining_features) == 0:
        # print 'stop condition 2 reached.'
        return create_leaf(target_values, data_weights)

    # stop condition 3: max_depth reached:
    if current_depth >= max_depth:
        # print 'stop condition 3 reached.'
        return create_leaf(target_values, data_weights)

    best_feature = best_splitting_feature(data, remaining_features, target, data_weights)

    if best_feature is None:
        # print 'data is identical in each feature.'
        return create_leaf(target_values, data_weights)

    remaining_features.remove(best_feature)

    yes_part = data[data[best_feature] == 1]
    no_part = data[data[best_feature] == 0]

    yes_indices = get_indices_from_mask(data[best_feature] == 1)
    no_indices = get_indices_from_mask(data[best_feature] == 0)

    # print "Split on feature %s. (%s, %s)" % (best_feature, len(yes_part), len(no_part))

    # if is perfect split, create a leaf
    if (len(yes_part) == len(data)) or (len(no_part) == len(data)):
        # print "Create a leaf node on a perfect split"
        return create_leaf(target_values, data_weights)

    left_subtree = weighted_decision_tree_create(yes_part, remaining_features, target,
                                                 data_weights[yes_indices], current_depth+1, max_depth)
    right_subtree = weighted_decision_tree_create(no_part, remaining_features, target,
                                                  data_weights[no_indices], current_depth+1, max_depth)

    node = Node(False, None, left_subtree, right_subtree, best_feature)

    return node


def count_nodes(tree):
    if tree.is_leaf:
        return 1
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)


# create the classfier using decision tree
# x is a single data point
def classify(tree, x, annotate=False):
    # if tree['is_leaf']:
    if tree.is_leaf:
        if annotate:
            print "At leaf, predicting %s" % tree.prediction
        return tree.prediction
    else:
        split_feature_value = x[tree.splitting_feature]
        if annotate:
            print "Split on %s = %s" % (tree.splitting_feature, split_feature_value)
        if split_feature_value == 1:
            return classify(tree.left, x, annotate)
        else:
            return classify(tree.right, x, annotate)


# target: col_name
def evaluate_classification_error(tree, data, target):
    prediction = data.apply(lambda x: classify(tree, x), axis=1)
    num_mistakes = (prediction != data[target]).sum()
    return float(num_mistakes)/len(data)



# return the list of stumps with its coefficients
def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):

    alpha = np.ones(len(data), dtype=float) / len(data)
    coeffs = []  # coefficient for each stump
    stumps = []
    target_value = data[target]

    for t in range(num_tree_stumps):

        # Learn a weighted decision tree stump with max_depth = 1
        tree_stump = weighted_decision_tree_create(data, features, target,
                                                   data_weights=alpha, max_depth=1)
        stumps.append(tree_stump)

        prediction = data.apply(lambda x: classify(tree_stump, x), axis=1)

        mistake_mask = (prediction != data[target])
        mistake_indices = get_indices_from_mask(mistake_mask)
        correct_mask = (prediction == data[target])
        correct_indices = get_indices_from_mask(correct_mask)

        # since datapoint weights is normalized, weight_of_mistakes is weighted_error
        weighted_error = alpha[mistake_indices].sum()

        coeff_t = .5 * log(1 / weighted_error - 1)
        coeffs.append(coeff_t)

        # update datapoints weights
        alpha[correct_indices] *= exp(-coeff_t)
        alpha[mistake_indices] *= exp(coeff_t)

        alpha /= alpha.sum()

    return coeffs, stumps


def predict_adaboost(stump_weights, tree_stumps, data):
    scores = pd.Series(data=np.zeros(len(data)), dtype=float)
    for i, tree_stump in enumerate(tree_stumps):
        predictions = data.apply(lambda x: classify(tree_stump, x), axis=1)
        for j, pred in enumerate(predictions):
            scores[j] += pred
    return scores.apply(lambda x: +1 if x > 0 else -1)


stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, feature_list,
                                                       target, num_tree_stumps=10)

# predictions = predict_adaboost(stump_weights, tree_stumps, test_data)
# accuracy = (predictions == test_data[target]).sum() / float(len(test_data))
# print 'Accuracy of 10-component ensemble = %s' % accuracy
# print stump_weights


def print_stump(tree):
    split_name = tree.splitting_feature  # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree.prediction
        return None
    split_feature, split_value = split_name.split('=')
    print '                       root'
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]{1}[{0} == 1]    '.format(split_name, ' '*(27-len(split_name)))
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                 (%s)' \
        % (('leaf, label: ' + str(tree.right.prediction) if tree.right.is_leaf else 'subtree'),
           ('leaf, label: ' + str(tree.left.prediction) if tree.left.is_leaf else 'subtree'))

# print_stump(tree_stumps[0])
# print_stump(tree_stumps[1])


# this may take a while...
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data,
                                                       feature_list, target, num_tree_stumps=30)

train_error_all = []
for n in xrange(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], train_data)
    accuray = (predictions == train_data[target]).sum() / float(len(train_data))
    train_error_all.append(1.0 - accuray)
    print "Iteration %s, training error = %s" % (n, train_error_all[n-1])

test_error_all = []
for n in xrange(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], test_data)
    accuray = (predictions == test_data[target]).sum() / float(len(test_data))
    test_error_all.append(1.0 - accuray)
    print "Iteration %s, test error = %s" % (n, test_error_all[n-1])


plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1, 31), train_error_all, '-', linewidth=4.0, label='Training error')
plt.plot(range(1, 31), test_error_all, '-', linewidth=4.0, label='Test error')

plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})
plt.legend(loc='best', prop={'size': 15})
# plt.tight_layout()
plt.show()
