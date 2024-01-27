import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle


def algorithm4(A, B, C, Y, UB, VBT, UC, VCT, dividend):
    Ap = A - B @ Y @ C
    At = UB.T @ Ap @ VCT.T 
    return VBT.T @ np.divide(At, dividend) @ UC.T + Y
