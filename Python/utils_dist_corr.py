import pandas as pd
import numpy as np
from scipy import stats
from itertools import product, combinations

subjects_no_out = ['S29-SBC', 'S17-CSB', 'S18b-SBC', 'S03-BSC', 'S25-BCS', 'S17-BCS', 'S01-BCS',
                        'S24-SCB', 'S11-SBC', 'S01-BSC', 'S30-SCB', 'S12-CBS', 'S31-CSB', 'S23-SBC',
                        'S11b-SBC', 'S13-BCS', 'S05-BCS', 'S20-SCB', 'S14-CBS', 'S12-SCB', 'S04-SCB',
                        'S07-BSC', 'S07-BCS', 'S15-SBC', 'S19-BCS', 'S26-BSC', 'S08-CSB', 'S19-CBS',
                        'S16-CSB', 'S08-BSC', 'S18-SBC', 'S02-CSB', 'S16-BSC', 'S21-CBS', 'S27-CBS',
                        'S13-SCB', 'S28-CSB', 'S15-CBS', 'S03-CBS', 'S09-CBS', 'S21b-CBS', 'S06-CSB',
                        'S10-CSB', 'S33-CBS', 'S23-SCB', 'S20b-SCB', 'S22-CSB', 'S10-BCS']

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

def define_pairs_diff(names):
    """
    compute all pairs of stims from different category
    """
    all_pairs = [(p0, p1) for p0, p1 in combinations(names,2)]
    pairs = [p for p in all_pairs if p[0][0] != p[1][0]]
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