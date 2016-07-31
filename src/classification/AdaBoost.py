# train a decision tree, one-hot encoding
# use graphviz to visulize decision tree
# tree pruning

import numpy as np
import pandas as pd
import json
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
import math
import Queue

pd.options.mode.chained_assignment = None  # default='warn'

loans = pd.read_csv('../../data/lending_club_data.csv', low_memory=False)

loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']


target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
            'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
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
print "Number of safe loans  : %s" % len(safe_loans)
print "Number of risky loans : %s" % len(risk_loans)

# now the load_data is balanced
loans_data = risk_loans.append(safe_loans)
exit()

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
def encode_one_hot(dataframe, features):

    df = pd.DataFrame(dataframe[features])  # make a copy of original dataframe

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

    one_hot = pd.DataFrame(dvec.fit_transform(df.to_dict(orient='record')).toarray())
    one_hot.columns = dvec.get_feature_names()
    one_hot.index = df.index
    return one_hot


onehot_data = encode_one_hot(loans_data, features)
onehot_features = onehot_data.columns
onehot_data[target] = loans_data[target]

train_data, test_data = train_test_split(onehot_data, test_size=.2, random_state=0)