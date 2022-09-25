import os
import numpy as np
import pandas as pd
import qpsolvers
import matplotlib.pyplot

# Loading Data
# getting working directory path, Method 1: os.path.dirname(os.path.abspath(__file__)), Method 2: os.getcwd()
dir_path = os.getcwd()
pd_data_balanced = pd.read_csv(
    dir_path+'//Data//500 Balanced.csv', header=None, low_memory=False)
pd_data_imbalanced = pd.read_csv(
    dir_path+'//Data//500 ImBalanced.csv', header=None, low_memory=False)

pd_data_all = pd.concat([pd_data_balanced , pd_data_imbalanced], axis=0)

# Converting panda dataframe to numpy array
np_data_all = pd_data_all.to_numpy()

# Defining input X and output Y
X = np_data_all[0:1000, 2:4]
Y = np_data_all[0:1000, 8]

matplotlib.pyplot.scatter(X[0:500, 0],
                          X[0:500, 1], c='r', marker='.')
matplotlib.pyplot.scatter(
    X[500:1000, 0], X[500:1000, 1], c='b', marker='.')
matplotlib.pyplot.show()

print("done")
