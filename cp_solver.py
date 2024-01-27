def cp_positive_semidefinite(A,B,C,e=1e-16,a=10):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column),symmetric = True)
    obj=cp.Minimize(cp.norm(A-B@X@C, "fro")) 
    prob=cp.Problem(obj,[X >> 0])
    start = timeit.default_timer()
    prob.solve(solver=cp.SCS,eps_rel=e,eps_abs=e, acceleration_interval=a)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_non_negative(A,B,C,e=1e-16):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column))
    obj=cp.Minimize(cp.norm(A-B@X@C, "fro")) 
    prob=cp.Problem(obj,[X >= 0])
    start = timeit.default_timer()
    prob.solve(solver=cp.ECOS, feastol = e, reltol=e, abstol=e)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_symmetric(A,B,C,e=1e-16):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column), symmetric=True)
    obj=cp.Minimize(cp.norm(A-B@X@C, "fro")) 
    prob=cp.Problem(obj)
    start = timeit.default_timer()
    prob.solve(solver=cp.ECOS, feastol = e)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_toeplitz(A,B,C,e=1e-16):
    n=B.shape[1]
    X=cp.Variable((n,n))
    obj=cp.Minimize(cp.norm(A-B@X@C, "fro"))
    constraints = []
    for d in range(-n+1,n):
        for i in range(max(0,-d),min(n,n-d)):
            constraints.append(X[i][i+d]==X[max(0,-d)][max(0,-d)+d])
                
    prob=cp.Problem(obj,constraints)        
    start = timeit.default_timer()
    prob.solve(solver=cp.ECOS, feastol = e)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_correlation(A,B,C,e=1e-16,a=10):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column),PSD=True)
    obj=cp.Minimize(cp.norm(A-B@X@C, "fro"))
    constraints = [cp.diag(X)==np.ones(row)]
    prob=cp.Problem(obj,constraints)  
    start = timeit.default_timer()
    prob.solve(solver=cp.SCS,eps_rel=e,eps_abs=e, acceleration_interval=a)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_stochastic(A,B,C,e=1e-16):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column))
    obj=cp.Minimize(cp.norm(A-B@X@C, "fro")) 
    prob=cp.Problem(obj,[X >= 0, X @ np.ones(row) == np.ones(row)])
    start = timeit.default_timer()
    prob.solve(solver=cp.ECOS, feastol = e, reltol=e, abstol=e)
    stop = timeit.default_timer()
    return X.value, stop-start
