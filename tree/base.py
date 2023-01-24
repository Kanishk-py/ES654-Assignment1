"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index

np.random.seed(42)

# Class for nodes in the decision tree
@dataclass
class Node:
    """
    Class to represent a node in the decision tree
    """
    classification: Literal["discrete", "continuous"] = None
    child_dict: dict = field(default_factory=dict) 
    attr: int = None
    position: int = None
    value: float = None

    # Function to print the Decision Tree
    def printTree(self, gap: int = 4) -> None:
        # Base Case
        if self.attr == None:
            print("{:.4f}".format(self.value))
            return

        for i in self.child_dict:

            if self.classification == "discrete":
                print(f"{' ' * (gap-4)}?(X({self.attr}) = {i}):")
                if self.child_dict[i].attr == None:
                    print(f"{' ' * gap}", end=" ")

            elif self.classification == "continuous":
                ans = "Y" if i == "right" else "N"

                if ans == "Y":
                    print(f"?(X({self.attr}) > {'{:.4f}'.format(self.position)}):")

                print(f"{' ' * gap}{ans}:", end=" ")
            # Recursively call the function
            self.child_dict[i].printTree(gap+4)

    # Function to recursively plot the Decision Tree
    def getVal(self, X: pd.DataFrame) -> float:
        if self.attr == None:
            return self.value

        if self.classification == "discrete":
            return self.child_dict[X[self.attr]].getVal(X)
        elif self.classification == "continuous":
            if X[self.attr] <= self.position:
                return self.child_dict["left"].getVal(X)
            else:
                return self.child_dict["right"].getVal(X)

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int = np.inf # The maximum depth the tree can grow to
    root: "Node" = None

    def build(self, X: pd.DataFrame, y: pd.Series, depth: int=0) -> Node:
        """
        Function to construct the decision tree recursively
        """

        # Base Case for recursion
        if y.unique().size == 1:
            return Node(value=y.unique()[0])

        # Checking conditions to stop recursion
        if len(X.columns) > 0 and depth < self.max_depth and len(list(X.columns)) != sum(list(X.nunique())):

            max_info_gain = -np.inf
            possible_split = None # This will be used for Real Input
            split_attr = None # This will be used to store the attribute to split on

            # Loop over all the attributes to find the best attribute to split on
            for attr in list(X.columns):
                info_gain = information_gain(y, X[attr], self.criterion)
                
                # If the attribute is continuous, we need to store the possible split position
                if X[attr].dtypes.name != "category":
                    info_gain, possible_split = info_gain[0], info_gain[1]
                
                # If the information gain is greater than the maximum information gain, 
                # update the maximum information gain and the split position
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    split_position = possible_split
                    split_attr = attr

            # Create a node
            currNode = Node(attr=split_attr)
            # Extract the column to split on
            splitted_col = X[split_attr]
            if X[split_attr].dtypes.name == "category":
                currNode.classification = "discrete"
                # Loop over all discrete values to create child nodes
                for i in splitted_col.unique():
                    currNode.child_dict[i] = self.build(X[splitted_col == i].drop(split_attr, axis=1), y[splitted_col == i], depth+1)
            else:
                # Create left and right child nodes
                currNode.classification = "continuous"
                currNode.child_dict["right"] = self.build(X[splitted_col > split_position].drop(split_attr, axis=1), y[splitted_col > split_position], depth+1)
                currNode.child_dict["left"] = self.build(X[splitted_col <= split_position].drop(split_attr, axis=1), y[splitted_col <= split_position], depth+1)
                currNode.position = split_position

            if y.dtypes.name == "category":
                currNode.value = y.mode()[0]
            else:
                currNode.value = y.mean()

            return currNode
        else:
            if y.dtypes.name == "category":
                return Node(value=y.mode()[0])
            else:
                return Node(value=y.mean())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.root = self.build(X, y)
                

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        Y = []
        for x in (X.index):
            Y.append(self.root.getVal(X.loc[x]))
        Y_hat = pd.Series(Y)
        return Y_hat

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.root.printTree()
