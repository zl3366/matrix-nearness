import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle


generator = {
  'psd' : generate_random_positive_semidefinite, 
    'nn'  : generate_random_non_negative,
    'sym' : generate_random_symmetric,
    'tp'  : generate_random_toeplitz,
    'st'  : generate_random_stochastic,
    'co'  : generate_random_correlation
}

cvx = {
  'psd' : cp_positive_semidefinite, 
    'nn'  : cp_non_negative,
    'sym' : cp_symmetric,
    'tp'  : cp_toeplitz,
    'st'  : cp_stochastic,
    'co'  : cp_correlation
}

projection = {
  'psd' : positive_semidefinite_projection, 
    'nn'  : non_negative_projection,
    'sym' : symmetric_projection,
    'tp'  : toeplitz_projection,
    'st'  : [row_sum_to_one, non_negative_projection],
    'co'  : [diagonal_to_one, positive_semidefinite_projection]
}




class problem:
    
    def __init__(self, ppty, n, gen_A='true', cond_P=0):
        self.ppty = ppty
        self.n = n
        self.gen_A=gen_A
        
        if cond_P == 0:
            self.B = generate_random(n, max(1,np.random.rand()*n))
            self.C = generate_random(n, max(1,np.random.rand()*n))
        else:
            self.B = generate_random(n, cond_P)
            self.C = generate_random(n, cond_P)
        
        if gen_A == 'random':
            self.A = generate_random(n, max(1,np.random.rand()*n))
        
        elif gen_A == 'perturb':
            self.X = generator[ppty](n, max(1,np.random.rand()*n))
            self.A = self.B @ self.X @ self.C + (np.random.rand(n,n)-0.5)*1e-8
        
        elif gen_A == 'true':
            self.X = generator[ppty](n, max(1,np.random.rand()*n*n))
            self.A = self.B @ self.X @ self.C



test_problem = problem(ppty='psd', n=32)
convergence_plot(test_problem, save=True, name = 'Accuracy (positive semidefinite matrix)')

test_problem = problem(ppty='co', n=32)
convergence_plot2(test_problem, save=True, name = 'Accuracy (correlation matrix)')

compare_speed('psd', 'true', name='Speed (positive semidefinite matrix)', save = True, tol=1e-4)
