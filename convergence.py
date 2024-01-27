import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle

def convergence_plot(problem, name='', save = False, iteration = 5000):
    
    ppty = problem.ppty
    n = problem.n
    A=problem.A
    B=problem.B
    C=problem.C
    X_true=problem.X
    
    
    UB, sb, VBT = la.svd(B)
    UC, sc, VCT = la.svd(C)
    tt1 = sb[:,None]
    tt2 = sc[:,None]
    s = tt1 @ tt2.T
    lamda = tt1[0]*tt1[-1]*tt2[0]*tt2[-1]
    dividend = s + lamda * np.divide(np.ones((n,n)),s)
    U = generate_random(n,1)
    Z = generator[ppty](n, max(1,np.random.rand()*n))
    
    X_cp, cp_time = cvx[ppty](A,B,C)
    cvxerror = la.norm(X_cp - X_true, "fro")/la.norm(X_true, "fro")
    
    
    forward_error_list = []
    for i in range(iteration):
        X = algorithm4(A, B, C, Z-U, UB, VBT, UC, VCT, dividend)
        Z = projection[ppty](X+U)
        U = U + X - Z
        forward_error = la.norm(Z - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    

    
    fig, ax = plt.subplots(1,1, dpi=1200)
    ax.set_title(name)
    ax.set_ylabel('forward error')
    ax.set_xlabel('iteration')
    ax.set_yscale('log')
    plt.plot(forward_error_list, label='NLA (error per iteration)')
    ax.hlines(y=cvxerror, xmin=0, xmax=5000, color = 'red', label="CVX (final error)")
    ax.legend()
    plt.show()
    

    if save:
        with open(name,'wb') as fid:
            pickle.dump(ax, fid)
        ax.figure.savefig(name+'.png')


def convergence_plot2(problem, name='', save = False, iteration = 5000):
    
    ppty = problem.ppty
    n = problem.n
    A=problem.A
    B=problem.B
    C=problem.C
    X_true=problem.X
    
    
    UB, sb, VBT = la.svd(B)
    UC, sc, VCT = la.svd(C)
    tt1 = sb[:,None]
    tt2 = sc[:,None]
    s = tt1 @ tt2.T
    lamda = tt1[0]*tt1[-1]*tt2[0]*tt2[-1]
    dividend = s + lamda * np.divide(np.ones((n,n)),s)
    U = generate_random(n,1)
    V = generate_random(n,1)
    Z = generator[ppty](n, max(1,np.random.rand()*n))
    
    
    X_cp, cp_time = cvx[ppty](A,B,C)
    cvxerror = la.norm(X_cp - X_true, "fro")/la.norm(X_true, "fro")
    
    
    forward_error_list = []
    for i in range(iteration):
        X = algorithm4(A, B, C, Z-U, UB, VBT, UC, VCT, dividend)
        Y = projection[ppty][0](Z-V)
        Z = projection[ppty][1]((X+Y+U+V)/2.)
        U = U + X - Z
        V = V + Y - Z
        forward_error = la.norm(Z - X_true, "fro")/la.norm(X_true, "fro")
       
        forward_error_list.append(forward_error)

    

    
    fig, ax = plt.subplots(1,1, dpi=1200)
    ax.set_title(name)
    ax.set_ylabel('forward error')
    ax.set_xlabel('iteration')
    ax.set_yscale('log')
    plt.plot(forward_error_list, label='NLA (error per iteration)')
    ax.hlines(y=cvxerror, xmin=0, xmax=5000, color = 'red', label="CVX (final error)")
    ax.legend()
    plt.show()
    

    if save:
        with open(name,'wb') as fid:
            pickle.dump(ax, fid)
        ax.figure.savefig(name+'.png')
