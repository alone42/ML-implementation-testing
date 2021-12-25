import numpy as np
import pandas as pd

from scipy.sparse import isspmatrix

def freq(X, prob):
    unique_x, counts_unique_x = np.unique(X, return_counts = True)
    # if isspmatrix(X):
    #     print(unique_x)
    if prob==True:
        probability_x = counts_unique_x / np.full(counts_unique_x.size, X.size)
        return unique_x, probability_x
    return unique_x, counts_unique_x

def freq2(X, Y, prob):
    if isspmatrix(X):
        inter = np.intersect1d(X.indices, Y.indices)
        cross_tab = np.array([[X.shape[0]-(Y.indices.shape[0]+X.indices.shape[0]), Y.indices.shape[0]], [X.indices.shape[0], inter.shape[0]]])
        if prob:
            return cross_tab / np.full(cross_tab.shape, np.sum(cross_tab))
        return cross_tab
    crosstab = pd.crosstab(X, Y, rownames='X', colnames='Y')
    counts_unique_x_y = np.asarray(crosstab)
    if prob==True:
        probability_x_y = counts_unique_x_y / np.full(counts_unique_x_y.shape, np.sum(counts_unique_x_y))
        return probability_x_y
    return np.asarray(crosstab)

def kappa_fun(p):
    entropy = lambda p: -np.sum(p * np.log2(p, out=np.zeros_like(p), where=(p!=0)))
    infogain = lambda p: entropy(np.sum(p, axis=1)) + entropy(np.sum(p, axis=0)) - entropy(p) 
    kappa = lambda p: infogain(p) / entropy(np.sum(p, axis=0))
    return kappa(p)
    

def ginigain_fun(p):
    gini = lambda p: 1 - np.sum(p*p)
    gini_YIX = lambda p:(np.sum((np.sum(p, axis=1) * [gini(pi) for pi in (p/np.outer( np.sum(p, axis=1), np.ones(p.shape[1])))])))
    ginigain = lambda p: gini(np.sum(p, axis=0)) - gini_YIX(p)
    return ginigain(p)