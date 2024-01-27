import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle


def speed_test(problem, thres):
    
    
    ppty = problem.ppty
    n = problem.n
    A=problem.A
    B=problem.B
    C=problem.C
    gen_A = problem.gen_A
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
        
    
  
    
  
        
        
    ad_error = la.norm(Z - X_true, "fro")/la.norm(X_true, "fro")


    start = timeit.default_timer()
    while ad_error> thres:
        X = algorithm4(A, B, C, Z-U, UB, VBT, UC, VCT, dividend)
        Z = projection[ppty](X+U)
        U = U + X - Z
        ad_error = la.norm(Z - X_true, "fro")/la.norm(X_true, "fro")
    
    stop = timeit.default_timer()
        
   
    ad_time = stop - start
   
    return ad_time, ad_error


def cp_speed(problem, featol,a):
    ppty = problem.ppty
    n = problem.n
    A=problem.A
    B=problem.B
    C=problem.C
    gen_A = problem.gen_A
    X_true=problem.X
    if ppty == 'co' or ppty =='psd':
         X_cp, cp_time = cvx[ppty](A,B,C, featol, a)
    else:
         X_cp, cp_time = cvx[ppty](A,B,C, featol)
   
    cp_error = la.norm(X_cp - X_true, "fro")/la.norm(X_true, "fro")
    return cp_time, cp_error



def speed_test2(problem, thres):

    
    ppty = problem.ppty
    n = problem.n
    A=problem.A
    B=problem.B
    C=problem.C
    gen_A = problem.gen_A
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
        
    
        
    ad_error = la.norm(Z - X_true, "fro")/la.norm(X_true, "fro")
    
    start = timeit.default_timer()
    while ad_error > thres:
        X = algorithm4(A, B, C, Z-U, UB, VBT, UC, VCT, dividend)
        Y = projection[ppty][0](Z-V)
        Z = projection[ppty][1]((X+Y+U+V)/2.)
        U = U + X - Z
        V = V + Y - Z
        ad_error = la.norm(Z - X_true, "fro")/la.norm(X_true, "fro")
    
    stop = timeit.default_timer()
        
  
    ad_time = stop - start
    
    return ad_time, ad_row


def compare_speed(ppty, gen_A, name='', save = False, tol=1e-8, a = 1):
    
    ad_list = []
    cp_list = []
    
    
    for n in [2,4,8,16,32,64,128]:
        
        print(n)
        
        temp = problem(ppty, n=n, gen_A=gen_A)
        cp_time, cp_error = cp_speed(temp, tol,a)
        
        tol = tol * (n**2)  
        asum = 0
        for i in range(10):
            
            if ppty == 'co' or ppty =='st':
                ad, ad_error= speed_test2(temp, cp_error)
            else:
                ad, ad_error = speed_test(temp, cp_error)
            asum += ad
            
        
        cp_list.append(cp_time)
        ad_list.append(asum/10)
    

       
    
    
    fig, ax = plt.subplots(1,1, dpi=1200)
    ax.set_title(name)
    ax.set_ylabel('time/s')
    ax.set_xlabel('dimension')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    plt.plot([2,4,8,16,32,64,128], ad_list, label='NLA')
    plt.plot([2,4,8,16,32,64,128], cp_list, label='CVX')
    ax.legend()
    plt.show()
    

    if save:
        with open(name,'wb') as fid:
            pickle.dump(ax, fid)
        ax.figure.savefig(name+'.png')
