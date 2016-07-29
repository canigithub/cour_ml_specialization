
# train a decision tree, one-hot encoding

import numpy as np
import graphviz as gv
import pydot
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

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
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
# print "Number of safe loans  : %s" % len(safe_loans_raw)
# print "Number of risky loans : %s" % len(risk_loans_raw)

percentage = len(risk_loans_raw)/float(len(safe_loans_raw))
risk_loans = risk_loans_raw
safe_loans = safe_loans_raw.sample(frac=percentage, random_state=1)
# print "Number of safe loans  : %s" % len(safe_loans)
# print "Number of risky loans : %s" % len(risk_loans)


loans_data = risk_loans.append(safe_loans)


# base: the unit of rounding, e.g. 1, 10, 100 etc
def round_down_to(num, base):
    return int(math.floor(num / base)) * base


def get_num_length(num):
    num = abs(int(math.ceil(num)))
    return len(str(num))


# given the numpy list: unique_values (len > 10)
# return a set of categorical variables (maximum is 10)
# improvement: analysis the statistics before split the categ
def get_categorical_from_numerical(unique_values):
    categorical = []
    variable_range = unique_values.max() - unique_values.min()
    n = get_num_length(variable_range)
    base = 10**(n-1)
    value = unique_values.min()
    while value < unique_values.max():
        categorical.append(round_down_to(value, base))
        value += base

    return categorical

# one-hot encode
# dataframe: pd.Dataframe, features: LIST of column names
# the default to_dict() will do nothing to float value type
def encode_one_hot(dataframe, features):

    df = pd.DataFrame(dataframe[features])  # make a copy of original dataframe

    # split the numerical feature if its unique values are too many
    for feat in features:
        if (df[feat].dtypes == int or df[feat].dtypes == float) \
                and len(df[feat].unique()) > 10:
            unique_values = df[feat].unique()
            categorical_variables = get_categorical_from_numerical(unique_values)

            for i, value in enumerate(categorical_variables):
                name = ''
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

    dvec = DictVectorizer()

    # orient='record' creates a list of dicts where each dict represents each row
    one_hot = pd.DataFrame(dvec.fit_transform(
        df.to_dict(orient='record')).toarray())
    one_hot.columns = dvec.get_feature_names()
    print dvec.get_feature_names()
    one_hot.index = df.index
    return one_hot


onehot = encode_one_hot(loans_data, features)



# train_data, valid_data = train_test_split(loans_data, test_size=.2, random_state=0)


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #
# use sklearn Decision Tree Classifier
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.* #

# decision_tree_model = DecisionTreeClassifier(max_depth=6, random_state=0)
# small_model = DecisionTreeClassifier(max_depth=2, random_state=0)
# small_model.fit(train_data[features], train_data[target])
# export_graphviz(small_model)
