
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Read real-estate data set
# ...
# 
yX = pd.read_fwf("C:\\Users\\hii\\Desktop\\ML\\auto-mpg.data", header = None) 
yX.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]  
yX["horsepower"]  = yX["horsepower"].str.replace("?", "NaN", regex= False).astype(float)
yX["horsepower"].fillna(value= yX["horsepower"].mean(), inplace=True) 


X = yX.iloc[:, 1:8] 
y = pd.Series(yX.iloc[:, 0])

# print(yX) 
# print(X) 
# print(y)

X_train, X_test = np.split(X, [int(X.shape[0] * 0.7)])
X_train = pd.DataFrame(X_train) 
X_test = pd.DataFrame(X_test)

y_train, y_test = np.split(y, [int(y.shape[0] * 0.7)])
y_train = pd.Series(y_train, dtype= "float64") 
y_test = pd.Series(y_test, dtype= "float64")
y_test = y_test.reset_index(drop=True)
# print(X_train) 
# print(X_test) 
# print(y_train)
# print(y_test)
# max_depth = 5 
tree = DecisionTree(criterion= "information_gain")
tree.fit(X_train, y_train) 

# tree.plot() 
y_hat = tree.predict(X_test)  
summary = {"Metric": ["RMSE", "MAE"], "Our_model":[], "Scikit":[]} 
print(y_hat-y_test)
tree_sk = DecisionTreeRegressor(random_state=0)
tree_sk.fit(X_train, y_train)
y_hat_sk = pd.Series(tree_sk.predict(X_test)) 
summary["Our_model"].append(rmse(y_hat, y_test)) 
summary["Our_model"].append(mae(y_hat, y_test)) 
summary["Scikit"].append(rmse(y_hat_sk, y_test)) 
summary["Scikit"].append(mae(y_hat_sk,  y_test)) 

print(pd.DataFrame(summary))




