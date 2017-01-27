import pandas as pd
import math
import numpy as np

# read data sets
def dataset_read(train_path, test_path):
    # Load training data and assign column names
    train_path, test_path = "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train","http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test"
    train = pd.read_table(train_path,
                          sep=" ", header=None)
    train = train.drop([0, 8], 1)
    col_names = ["class", "a1", "a2", "a3", "a4", "a5", "a6"]
    train.columns = col_names

    test = pd.read_table(test_path,
                         sep=" ", header=None)
    test = test.drop([0, 8], 1)
    test.columns = col_names

    ## Convert the columns into category
    for i in range(len(col_names)):
        train[col_names[i]] = train[col_names[i]].astype("category")
        test[col_names[i]] = test[col_names[i]].astype("category")
    return train,test