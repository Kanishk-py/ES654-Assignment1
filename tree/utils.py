import pandas as pd
import numpy as np
from typing import Literal


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value, counts = np.unique(Y, return_counts=True)
    prob = counts / counts.sum()
    entropy = 0
    for p in prob:
        entropy -= p * np.log2(p)

    return entropy

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    value, counts = np.unique(Y, return_counts=True)
    prob = counts / counts.sum()
    gini = 1
    gini -= np.sum(prob ** 2)

    return gini


def information_gain(Y: pd.Series, attr: pd.Series, criterion: Literal["information_gain", "gini_index"]) -> float:
    """
    Function to calculate the information gain
    """

    # Discrete Input and Discrete Output
    if (Y.dtypes.name == "category" and attr.dtypes.name == "category"):
        info_gain = entropy(Y) if criterion == "information_gain" else gini_index(Y)
        classes = attr.unique()
        for cls in classes:
            info_gain -= ((attr == cls).sum() / attr.size) * ( entropy(Y[attr == cls]) if criterion == "information_gain" else gini_index(Y[attr == cls]) )

        return info_gain
    
    # Real Input and Discrete Output
    elif (Y.dtypes.name == "category" and attr.dtypes.name == "float64"):
        dataset = pd.concat([Y, attr], axis=1, join='inner')
        dataset.columns = ['Y', 'attr']
        dataset.sort_values(by=['attr', 'Y'], inplace=True)

        Attr = dataset['attr'].to_numpy().flatten()
        Y_new = dataset['Y'].to_numpy().flatten()

        max_info_gain = 0
        split = 0

        total_entropy = entropy(Y_new) if criterion == "information_gain" else gini_index(Y_new)

        for i in range(1, len(Attr)):
            if (Attr[i] != Attr[i-1]):
                info_gain = total_entropy
                info_gain -= i / len(Attr) * ( entropy(Y_new[:i]) if criterion == "information_gain" else i / len(Attr) * gini_index(Y_new[:i]) )
                info_gain -= (len(Attr) - i) / len(Attr) * ( entropy(Y_new[i:]) if criterion == "information_gain" else (len(Attr) - i) / len(Attr) * gini_index(Y_new[i:]) )
                if max_info_gain < info_gain:
                    max_info_gain = info_gain
                    split = float( Attr[i] + Attr[i-1] ) / 2

        return max_info_gain, split

    # Real Input and Real Output
    elif (Y.dtypes.name == "float64" and attr.dtypes.name == "float64"):
        dataset = pd.concat([Y, attr], axis=1, join='inner')
        dataset.columns = ['Y', 'attr']
        dataset.sort_values(by=['attr', 'Y'], inplace=True)

        Attr = dataset['attr'].to_numpy().flatten()
        Y_new = dataset['Y'].to_numpy().flatten()

        max_info_gain = 0
        split = 0

        total_var = np.var(Y_new)

        for i in range(1, len(Attr)):
            if (Attr[i] != Attr[i-1]):
                info_gain = total_var
                info_gain -= i / len(Attr) * np.var(Y_new[:i])
                info_gain -= (len(Attr) - i) / len(Attr) * np.var(Y_new[i:])
                if max_info_gain < info_gain:
                    max_info_gain = info_gain
                    split = float( Attr[i] + Attr[i-1] ) / 2

        return max_info_gain, split

    # Discrete Input and Real Output
    elif (Y.dtypes.name == "float64" and attr.dtypes.name == "category"):
        info_gain = np.var(Y)
        classes = attr.unique()
        for cls in classes:
            info_gain -= ((attr == cls).sum() / attr.size) * np.var(Y[attr == cls])
        return info_gain
