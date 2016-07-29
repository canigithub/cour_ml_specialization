
# train a decision tree

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

loans = pd.read_csv('../../data/lending_club_data.csv')

