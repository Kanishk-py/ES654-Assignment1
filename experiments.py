import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from typing import Union, Literal

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

# create a dataset with N samples and M features
def create_dataset(N: int, M: int, type: Literal['RR', 'DD', 'RD', 'DR']):

	if type[0] == 'R':
		X = pd.DataFrame(np.random.randn(N, M))
	else:
		X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(5)})

	if type[1] == 'R':
		y = pd.Series(np.random.randn(N))
	else:
		y = pd.Series(np.random.randint(2, size = N), dtype="category")
	
	return X, y

