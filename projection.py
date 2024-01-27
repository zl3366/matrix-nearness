import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle


def positive_semidefinite_projection(a):
    ah = (a+a.T)/2
    u, h = polar(ah)
    return (h+ah)/2

def non_negative_projection(a):
    return np.maximum(0, a)

def symmetric_projection(a):
    return (a+a.T)/2

def toeplitz_projection(a):
    n = a.shape[0]
    for i in range(0,n):
        right = np.mean(np.diagonal(a, offset=i))
        np.fill_diagonal(a[:,i:], right)
        left = np.mean(np.diagonal(a, offset=-i))
        np.fill_diagonal(a[i:,:], left)
    return a

def diagonal_to_one(a):
    b=a
    np.fill_diagonal(b, 1)
    return b

def row_sum_to_one(a):
    n = a.shape[0]
    return a - (a @ np.ones((n,n))-1)/n
    
