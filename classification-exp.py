import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)


#2a   *************************************************************************#
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

#Splitting X and y (numpy arrays) into two parts in ratio 7:3 and making it a pandas dataframe
X_train, X_test = np.split(X, [int(X.shape[0] * 0.7)])
X_train = pd.DataFrame(X_train) 
X_test = pd.DataFrame(X_test)

y_train, y_test = np.split(y, [int(y.shape[0] * 0.7)])
y_train = pd.Series(y_train, dtype= "category") 
y_test = pd.Series(y_test, dtype= "category")
# print(X) 
# print(y) 
# print(X_train) 
# print(X_test) 
# print(y_train)
# print(y_test)


for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) 
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    print("Decision Tree for {} is:".format(criteria))
    print()
    tree.plot()
    print()
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print(f'Precision({cls}): ', precision(y_hat, y_test, cls))
        print(f'Recall({cls}): ', recall(y_hat, y_test, cls))
    print()


#2b   *************************************************************************#

# 5 Fold cross validation 
print("==============================")
print("2 (b) : 5 fold cross-validation") 
print()
begin_test = 0
data = {"Fold": [], "Accuracy" : []}
for i in range(5):
    X_test_kf = pd.DataFrame(X[begin_test:begin_test + int(0.2*X.shape[0])] )
    y_test_kf = pd.Series(y[begin_test: begin_test + int(0.2*y.shape[0])] , dtype= "category")
    X_train_kf = pd.DataFrame(np.concatenate((X[0: begin_test] , X[begin_test + int(0.2*X.shape[0]) : ])))
    y_train_kf = pd.Series(np.concatenate((y[0:begin_test] , y[begin_test + int(0.2*y.shape[0]):] ) ), dtype= "category")
    # print(X_train_kf) 
    # print(X_test_kf) 
    # print(y_train_kf)
    # print(y_test_kf)
    tree = DecisionTree(criterion= "information_gain")
    tree.fit(X_train_kf, y_train_kf) 
    y_hat = tree.predict(X_test_kf) 
    data["Fold"].append(i+1) 
    data["Accuracy"].append( accuracy(y_hat, y_test_kf))
    # print("Accuracy for {}th fold is {}".format(i, accuracy(y_hat, y_test_kf)))  


    begin_test += int(X.shape[0]*0.2)   #Each fold/part will be 20% of the whole dataset

print(pd.DataFrame(data)) 
# Nested cross validation 
print()
begin_valid = 0

summary = {0:[], 1:[], 2:[], 3:[], 4:[]}
for i in range(5):
    X_valid = pd.DataFrame(X[begin_valid:begin_valid + int(0.2*X.shape[0])] )
    y_valid = pd.Series(y[begin_valid: begin_valid+ int(0.2*y.shape[0])] , dtype= "category")
    X_train_kf = pd.DataFrame(np.concatenate((X[0: begin_valid] , X[begin_valid + int(0.2*X.shape[0]) : ])))
    y_train_kf = pd.Series(np.concatenate((y[0:begin_valid] , y[begin_valid + int(0.2*y.shape[0]):] ) ) , dtype= "category")
    # Depths are varied from 0 to 7.
    for depth in range(8): 
        tree = DecisionTree(criterion= "information_gain", max_depth = depth) 
        tree.fit(X_train_kf, y_train_kf) 
        y_hat_valid = tree.predict(X_valid) 
        summary[i].append(accuracy(y_hat_valid, y_valid))      #storing the accuracy for each depth
    begin_valid += int(X.shape[0]*0.2)

summary = pd.DataFrame(summary)
summary["Average Score"] = summary.mean(axis=1) 
print(summary)
print()


#The optimal depth corrosponds to that at which the avg of accuracy scores of all folds is maximum.
print("Nested Cross Validation using 5 folds:")
print("The optimal depth is {} and it has accuracy = {}".format(summary["Average Score"].idxmax(), round(summary["Average Score"].max(), 2)))