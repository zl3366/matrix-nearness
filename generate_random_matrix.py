import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle


def generate_random_orthogonal(n): 
    U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    return U

def generate_random(n, cond_P): 
    log_cond_P = np.log(cond_P)
    s = np.exp((np.random.rand(n)-0.5) * log_cond_P)
    S = np.diag(s)
    U = generate_random_orthogonal(n)
    V = generate_random_orthogonal(n)
    P = U.dot(S).dot(V.T)
    return P

def generate_random_symmetric(n, cond_P):
    temp = generate_random(n, cond_P)
    return (temp + temp.T)/2

def generate_random_non_negative(n, cond_P):
    return np.maximum(generate_random(n, cond_P), 0)

def generate_random_positive_semidefinite(n, cond_P=1):
    temp = generate_random(n, cond_P)
    return temp @ temp.T

def generate_random_correlation(n, cond_P):
    temp = generate_random_positive_semidefinite(n, cond_P)
    s = np.diag(temp)
    x = np.diag(np.divide(np.ones(n),np.sqrt(s)))
    return x @ temp @ x

def generate_random_toeplitz(n, cond_P):
    temp = generate_random(n, cond_P)
    return toeplitz_projection(temp)

def generate_random_stochastic(n, cond_P):
    temp = generate_random_non_negative(n, cond_P)
    return np.divide(temp, temp @ np.ones((n,n)))
