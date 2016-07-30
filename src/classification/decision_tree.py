
# train a decision tree, one-hot encoding
# use graphviz to visulize decision tree

import numpy as np
import pandas as pd
import json
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
import math

pd.options.mode.chained_assignment = None  # default='warn'

loans = pd.read_csv('../../data/lending_club_data.csv', low_memory=False)

loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']


# less features for the second part
features = ['grade',                     # grade of the loan
            # 'sub_grade',                 # sub-grade of the loan
            # 'short_emp',                 # one year or less of employment
            # 'emp_length_num',            # number of years of employment (for part 1)
            'emp_length',                # number of years of employment (for part 2)
            'home_ownership',            # home_ownership status: own, mortgage or rent
            # 'dti',                       # debt to income ratio
            # 'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            # 'last_delinq_none',          # has borrower had a delinquincy
            # 'last_major_derog_none',     # has borrower had 90 day or worse rating
            # 'revol_util',                # percent of available credit being used
            # 'total_rec_late_fee',        # total late fees received to day
            ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

# split data to train and validation
with open('../../data/module_5_assignment_1_train_idx.json') as f:
    train_idx = json.load(f)

with open('../../data/module_5_assignment_1_validation_idx.json') as f:
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


# base: the unit of rounding, e.g. 1, 10, 100 etc
def round_down_to(num, base):
    return int(math.floor(num / base)) * base


def get_num_length(num):
    num = abs(int(math.ceil(num)))
    return len(str(num))


# given the numpy list: unique_values (len > 10)
# return a set of categorical variables (maximum is 10)
# improvement: analysis the statistics before split the categ
# for example, total_rec_late_fee column is highly skewed. for this
# kind of column, better divide by it's density
def numerical_to_categorical(unique_values):
    categorical = []
    variable_range = unique_values.max() - unique_values.min()
    n = get_num_length(variable_range)
    base = 10**(n-1)
    value = unique_values.min()
    while value < unique_values.max():
        categorical.append(round_down_to(value, base))
        value += base

    return categorical


# one-hot encoder: for numerical values, if unique values > 10, then use a set of
# ranges to represent it. for string values, represent each value individually.
# dataframe: pd.Dataframe, features: LIST of column names
# !!!For practical use, it's crucial improve the numerical_to_categorical function
# to divide the values more resonablely.!!!
# (for example, based on the distribution density)
def encode_one_hot(dataframe, features):

    df = pd.DataFrame(dataframe[features])  # make a copy of original dataframe

    # split the numerical feature if its unique values are too many
    # if unique values <= 10, just use the values to split
    for feat in features:
        if (df[feat].dtypes == int or df[feat].dtypes == float) \
                and len(df[feat].unique()) > 10:
            unique_values = df[feat].unique()
            categorical_variables = numerical_to_categorical(unique_values)
            for i, value in enumerate(categorical_variables):
                if i == 0:
                    name = feat + ' < ' + str(value)
                    df[name] = df[feat].apply(lambda x: 1 if x < value else 0)
                elif i == len(categorical_variables) - 1:
                    name = feat + ' > ' + str(value)
                    df[name] = df[feat].apply(lambda x: 1 if x > value else 0)
                else:
                    val = categorical_variables[i-1]
                    name = str(val) + ' <= ' + feat + ' < ' + str(value)
                    df[name] = df[feat].apply(lambda x: 1 if val <= x < value else 0)

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

train_data, test_data = train_test_split(onehot_data, test_size=.2, random_state=0)


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use sklearn Decision Tree Classifier
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #


decision_tree_model = DecisionTreeClassifier(max_depth=6, random_state=0)
decision_tree_model.fit(train_data[onehot_features], train_data[target])
small_model = DecisionTreeClassifier(max_depth=2, random_state=0)
small_model.fit(train_data[onehot_features], train_data[target])
big_model = DecisionTreeClassifier(max_depth=10, random_state=0)
big_model.fit(train_data[onehot_features], train_data[target])

# export_graphviz(small_model, out_file='/Users/gerald/Desktop/tree.dot', feature_names=onehot_features)


def get_classification_accuracy(model, data, output):
    predictions = model.predict(data)
    num_correct = (predictions == output).sum()
    return float(num_correct) / len(output)

# print get_classification_accuracy(small_model, valid_data[onehot_features], valid_data[target])


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# Implementing Binary Decision Tree Clssifier from Scratch
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

# check point
# print "Total number of grade.A loans : %s" % onehot_data['grade=A'].sum()
# print "Expexted answer               : 6422"  # error comes from train_test_split


# use majority class classification
# all data points that are not in the majority class are considered mistakes
# keep in mid that the target label is 1 or -1, but feature label is 1 or 0
def intermediate_node_num_mistakes(labels_in_node):
    if len(labels_in_node) == 0:
        return 0

    num_pos = (labels_in_node == +1).sum()
    num_neg = (labels_in_node == -1).sum()

    if num_pos >= num_neg:
        return num_neg
    else:
        return num_pos


# select the feature with lowest classification error
# if classification error is same, split on the lowest imbalance splitting
# all features are binary features
def best_splitting_feature(data, features, target):

    best_feature = None
    min_num_mistakes = float('inf')
    min_imbalance = float('inf')  # measure the imbalance of splitted parts

    for feature in features:
        yes_part = data[data[feature] == 1]
        no_part = data[data[feature] == 0]

        num_err_yes = intermediate_node_num_mistakes(yes_part[target])
        num_err_no = intermediate_node_num_mistakes(no_part[target])

        # since total # of data points are all the same, thus can use #
        # of mistakes directly
        if (num_err_yes + num_err_no) < min_num_mistakes:
            min_num_mistakes = num_err_yes + num_err_no
            best_feature = feature
            min_imbalance = abs(len(no_part) - len(yes_part))
        elif (num_err_yes + num_err_no) == min_num_mistakes:
            imbalance = abs(len(no_part) - len(yes_part))
            if imbalance < min_imbalance:
                best_feature = feature
                min_imbalance = imbalance

    return best_feature


'''
start to build the decision tree. each node is represented as a dict:
    {
        'is_leaf': boolean,
        'prediction': prediction if at leaf node else None
        'left': left subtree
        'right': right subtree
        'splitting_feature': the feature that this node splits on
    }
'''


# create leaf node given a set of target values
def create_leaf(target_values):

    # set prediction to the majority class
    num_yes = (target_values == +1).sum()
    num_no = (target_values == -1).sum()

    prediction = +1 if num_yes > num_no else -1

    # create a leaf node
    leaf = {'is_leaf': True,
            'prediction': prediction,
            'left': None,
            'right': None,
            'splitting_feature': None
            }
    return leaf


# create decision tree with 3 terminate conditons:
# 1. all data points have same target value
# 2. not more features to split on
# 3. reach the max_depth
# yes -> go left, no -> go right
def decision_tree_create(data, features, target, current_depth=0, max_depth=10):

    remaining_features = features[:]  # Make a copy of the features
    target_values = data[target]

    # check if it's eligible to create a leaf node
    # case 1: if all data points has same value (mistakes == 0)
    num_mistakes = intermediate_node_num_mistakes(target_values)
    if num_mistakes == 0:
        # print 'Stop condition 1 reached'
        return create_leaf(target_values)

    # case 2: if no more features to split
    if len(remaining_features) == 0:
        # print 'Stop condition 2 reached'
        return create_leaf(target_values)

    # case 3: if max_depth is reached
    if current_depth >= max_depth:
        # print 'Stop condition 3 reached'
        return create_leaf(target_values)

    # otherwise, find the best feature to split on
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    remaining_features.remove(splitting_feature)

    # keep in mind the feature values are 1 and 0
    yes_part = data[data[splitting_feature] == 1]
    no_part = data[data[splitting_feature] == 0]
    # print 'Split on feature %s. (%s, %s)' % (splitting_feature, len(no_part), len(yes_part))

    # create a leaf node if the split is "perfect"
    # why can we create a leaf if the split is perfect? Because:
    # 1. split use any other feature will result in a worse error rate
    # 2. this is a perfect split, means the majority classification is best, means
    #    in the next recursion, no spliting will beat the majority classification
    #    (everything is the same except majority classification is prohibitted, cause
    #    you have to select one (worse) feature to split.) In general, next recursion
    #    will always be worse.
    # Thus, it's eligible to create a leaf here.
    if len(yes_part) == len(data) or len(no_part) == len(data):
        # print "Create a leaf node on a perfect split"
        return create_leaf(target_values)

    right_subtree = decision_tree_create(no_part, remaining_features, target,
                                         current_depth+1, max_depth)
    left_subtree = decision_tree_create(yes_part, remaining_features, target,
                                        current_depth+1, max_depth)

    node = {'is_leaf': False,
            'prediction': None,
            'left': left_subtree,
            'right': right_subtree,
            'splitting_feature': splitting_feature
            }

    return node


# count total nodes of a decision tree
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


feature_list = onehot_features

# if necessary cast to list type to support remove() function
if type(feature_list) == pd.indexes.base.Index:
    feature_list = feature_list.tolist()


# the test below fail. the reason is that while there are a lot features which have same
# classification error, but the order of them is different from the test present (the splitting
# feature will be first encounter first is order) which lead to different splits.
# small_data_decision_tree = decision_tree_create(train_data, feature_list, 'safe_loans', max_depth=3)
# if count_nodes(small_data_decision_tree) == 13:
#     print 'Test passed!'
# else:
#     print 'Test failed... try again!'
#     print 'Number of nodes found                :', count_nodes(small_data_decision_tree)
#     print 'Number of nodes that should be there : 13'


# create the classfier using decision tree
# x is a single data point
def classify(tree, x, annotate=False):
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 1:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


my_decision_tree = decision_tree_create(train_data, feature_list, target, max_depth=6)

# print test_data.iloc[0][target]
# prediction = classify(my_decision_tree, test_data.iloc[0])
# print prediction


def evaluate_classification_error(tree, data):
    # axis=1: apply to each row (in column direction)
    prediction = data.apply(lambda x: classify(tree, x), axis=1)
    num_mistakes = (prediction != data[target]).sum()
    return float(num_mistakes)/len(data)

# print evaluate_classification_error(my_decision_tree, test_data)


# print a single stump
def print_stump(tree, name='root'):
    split_name = tree['splitting_feature']  # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('=')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))


# print_stump(my_decision_tree)
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
