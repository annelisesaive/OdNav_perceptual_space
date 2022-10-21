import pandas as pd
import numpy as np
from scipy import stats
from itertools import product, combinations

def nb_pairs(n):
    """
    compute nb of potential pairs in an n dataset
    """
    nb = len([p for p in combinations(np.arange(n),2)])
    return nb

def define_pairs(names):
    """
    compute all pairs of 2 stim to compare
    """
    pairs = [(p0, p1) for p0, p1 in combinations(names,2)]
    return pairs

def compute_distance(A_xy, B_xy):
    """
    compute euclidian distance btw 2 points (A & B) using 2D coordinates
    """
    dist = np.round(np.linalg.norm(A_xy-B_xy),2)
    return dist

def compute_all_dist(df, pairs, name_col, x_col, y_col):
    """
    compute all euclidian distances btw all pairs
    """
    dist = np.zeros((len(pairs)))
    for i, p in enumerate(pairs):
        A_xy = df.loc[(df[name_col] == p[0])][[x_col,y_col]].values[0]
        B_xy = df.loc[(df[name_col] == p[1])][[x_col,y_col]].values[0]
        d = compute_distance(A_xy, B_xy)
        dist[i] += d
    return dist

def compute_all_R(df, pairs, meth='spearman'):
    """
    compute all correlations btw all pairs of data
    meth : method for computing correlations (spearman, pearson)
    """
    corr = np.zeros((len(pairs)))
    for i, p in enumerate(pairs):
        a1 = np.squeeze(df[[p[0]]].values)
        a2 = np.squeeze(df[[p[1]]].values)
        if meth == 'spearman':
            r = stats.spearmanr(a1, a2)[0]
        elif meth == 'pearson':
            r = stats.pearsonr(a1, a2)[0]
        else:
            print('unknown method for correlation')
        corr[i] += r
    return corr 