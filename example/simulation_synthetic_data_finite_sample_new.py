# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:35:17 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from scipy.spatial import distance
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import time

# =============================================================================
# Data
# =============================================================================

# Split to x, y, z
# feature = (x, z), target = (y)
d = 2    # dimension of x
q = 1    # dimension of z

# Covariance matrices
Cov_xzy = np.array([[1, .25, .25, .5],
                    [.25, 1, .25, .5],
                    [.25, .25, 1, .5],
                    [.5, .5, .5, 1]])
np.linalg.eig(Cov_xzy)
plt.figure(figsize=(6, 5))
sns.heatmap(Cov_xzy)

iter_max = 50

MSE_const = np.zeros((105, iter_max))
MSE_unconst = np.zeros((105, iter_max))
MSE_combine = np.zeros((105, iter_max))

for iter in range(iter_max):
    
    t = time.time()
    
    # Generated finite sample (size N) training data
    N = 100
    xzy = np.transpose(np.random.multivariate_normal(np.zeros(((d+q+1))), Cov_xzy, N))
    
    XZ = xzy[0:d+q, :]
    X = xzy[0:d, :].reshape(d, N)
    Z = xzy[d:d+q, :].reshape(q, N)
    Y = xzy[-1, :].reshape(1, N)
    
    D_j = np.zeros(d)
    for j in range (d):
        D_j[j] = np.sqrt(np.linalg.norm(X[j, :])**2/N)
    D = np.diag(D_j)
    
    M = np.dot(Z, np.transpose(X))
    M_dagger = np.linalg.pinv(M)
    
    B = Y - np.dot(np.transpose(np.dot(M_dagger, np.dot(Z, np.transpose(Y)))), X)
    A = np.dot(np.transpose(np.identity(d) - np.dot(M_dagger, M)), X)
    
    D_j = np.zeros(d)
    for j in range (d):
        D_j[j] = np.sqrt(np.linalg.norm(A[j, :])**2/N)
    D_new = np.diag(D_j)
    
    # =============================================================================
    # Training (Data fitting OLS-SPICE)
    # =============================================================================
    
    theta = cp.reshape(cp.Variable(d), (d, 1))
    
    def loss_fn_1(A, B, theta):
        return cp.norm(B - cp.matmul(cp.atoms.affine.transpose.transpose(theta), A)) 
    
    def regularizer_1(theta):
        return cp.norm(cp.matmul(D_new, theta), 1)
    
    def objective_fn_1(A, B, theta):
        return loss_fn_1(A, B, theta) + regularizer_1(theta)
    
#    def objective_fn_1(A, B, theta):
#        return loss_fn_1(A, B, theta)
    
    problem_1 = cp.Problem(cp.Minimize(objective_fn_1(A, B, theta)))
    problem_1.solve()
    
    # =============================================================================
    # Unconstrained optimal    
    # =============================================================================
    
    w_opt = cp.reshape(cp.Variable(d), (d, 1))
    
    def loss_fn_2(X, Y, w_opt):
        return cp.norm(Y - cp.matmul(cp.atoms.affine.transpose.transpose(w_opt), X)) 
    
    def regularizer_2(w_opt):
        return cp.norm(cp.matmul(D, w_opt), 1)
    
    def objective_fn_2(X, Y, w_opt):
        return loss_fn_2(X, Y, w_opt) + regularizer_2(w_opt)
    
#    def objective_fn_2(X, Y, w_opt):
#        return loss_fn_2(X, Y, w_opt)
    
    problem_2 = cp.Problem(cp.Minimize(objective_fn_2(X, Y, w_opt)))
    problem_2.solve()
    
    # =============================================================================
    # Predict z first, then y  
    # =============================================================================
    
    w_z = cp.reshape(cp.Variable(d), (d, 1))
    
    def loss_fn_3(X, Z, w_z):
        return cp.norm(Z - cp.matmul(cp.atoms.affine.transpose.transpose(w_z), X)) 
    
    def objective_fn_3(X, Z, w_z):
        return loss_fn_3(X, Z, w_z) + regularizer_2(w_z)
    
#    def objective_fn_3(X, Z, w_z):
#        return loss_fn_3(X, Z, w_z)

    problem_3 = cp.Problem(cp.Minimize(objective_fn_3(X, Z, w_z)))
    problem_3.solve()
    
    # =============================================================================
    # Testing
    # =============================================================================
    
    theta = theta.value[0:d]
    
    w_c = np.dot(M_dagger, np.dot(Z, np.transpose(Y))) + np.dot((np.identity(d) - np.dot(M_dagger, M)), theta)
    w_opt = w_opt.value[0:d]
    w_z = w_z.value
    
    # Threshold 
    T = 0.5
    K = 10
    
    # Generate testing data
    # (x, y, z) ~ N(mu, Cov)
    Cov_xyz = np.array([[1, .25, .5, .25,],
                        [.25, 1, .5, .25,],
                        [.5, .5, 1, .5],
                        [.25, .25, .5, 1]])
    #LA.eig(Cov_xyz)
    #plt.figure()
    #sns.heatmap(Cov_xyz)
    
    # (x, y | z) ~ N(mu_xy_z, Cov_xy_z)
    def conditional_mean(z, mu_z, mu_xy, Cov_xyz):
        mu_xy_z = mu_xy + np.dot(np.dot(Cov_xyz[0:d+1, d+1:], np.linalg.inv(Cov_xyz[d+1:, d+1:])), (z-mu_z))  
        Cov_xy_z = Cov_xyz[0:d+1, 0:d+1] - np.dot(np.dot(Cov_xyz[0:d+1, d+1:], np.linalg.inv(Cov_xyz[d+1:, d+1:])), np.transpose(Cov_xyz[0:d+1, d+1:]))
        return mu_xy_z, Cov_xy_z
    
    #mu_xy_z, Cov_xy_z = conditional_mean(1, 0, 0, Cov_xyz)
    num_test = 1000
    for j in range(0, 105, 1):
        z = -10 + j*.2
        mu_xy_z, Cov_xy_z = conditional_mean(z, 0, 0, Cov_xyz)
        y_true = np.zeros((num_test,1))
        y_predict_const = np.zeros((num_test,1))
        y_predict_unconst = np.zeros((num_test,1))
        z_predict = np.zeros((num_test,1))
        y_predict_combine = np.zeros((num_test,1))
        
        for i in range(0, num_test):
            xy = np.random.multivariate_normal(mu_xy_z.reshape(d+1), Cov_xy_z)
            
            y_true[i] = xy[d:]
            
            # Robust 
            y_predict_const[i] = np.dot(np.transpose(w_c), xy[0:d].reshape(d,1))
            
            # LMMSE
            y_predict_unconst[i] = np.dot(np.transpose(w_opt), xy[0:d].reshape(d,1))
            
            z_predict[i] = np.dot(np.transpose(w_z), xy[0:d].reshape(d,1))
                
            # Combine
            maha_distance = distance.mahalanobis(z_predict[i], 0, 1)
            a = 1/(1+np.exp(-K*(maha_distance-T)))
            b = 1 - a
            y_predict_combine[i] = a*y_predict_const[i] + b*y_predict_unconst[i]
            
        # MSE
        MSE_const[j, iter] = mean_squared_error(y_true, y_predict_const) 
        MSE_unconst[j, iter] = mean_squared_error(y_true, y_predict_unconst) 
        MSE_combine[j, iter] = mean_squared_error(y_true, y_predict_combine) 
        
    # do stuff
    elapsed = time.time() - t
    print(iter)
    print(elapsed)
        
MSE_const_average = np.mean(MSE_const, axis=1)
MSE_unconst_average = np.mean(MSE_unconst, axis=1)
MSE_combine_average = np.mean(MSE_combine, axis=1)

# =============================================================================
# Visualization of results    
# =============================================================================
rz = norm(loc=0, scale=1)
z = np.linspace(norm.ppf(0.0001),norm.ppf(0.9999), 100)    

x = np.array([3, 3])
mu_z_x = np.dot(np.dot(Cov_xzy[d:d+q, 0:d], np.linalg.inv(Cov_xzy[0:d, 0:d])), x.reshape(d,1))
var_z_x = 1 - np.dot(np.dot(Cov_xzy[d:d+q, 0:d], np.linalg.inv(Cov_xzy[0:d, 0:d])), np.transpose(Cov_xzy[d:d+q, 0:d]))
rz_x = norm(loc=mu_z_x, scale=var_z_x)
z_x = np.linspace(norm.ppf(0.0001),norm.ppf(0.9999), 100)  

MSE_const_var = np.std(MSE_const, axis=1)
MSE_unconst_var = np.std(MSE_unconst, axis=1)
MSE_combine_var = np.std(MSE_combine, axis=1)

MSE_const_median = np.median(MSE_const, axis=1)
MSE_unconst_median = np.median(MSE_unconst, axis=1)
MSE_combine_median = np.median(MSE_combine, axis=1)

MSE_const_75 = np.percentile(MSE_const, 75, axis=1)
MSE_unconst_75 = np.percentile(MSE_unconst, 75, axis=1)
MSE_combine_75 = np.percentile(MSE_combine, 75, axis=1)

MSE_const_25 = np.percentile(MSE_const, 25, axis=1)
MSE_unconst_25 = np.percentile(MSE_unconst, 25, axis=1)
MSE_combine_25 = np.percentile(MSE_combine, 25, axis=1)

plt.figure(figsize=(5, 3.5))
plt.fill_between(np.arange(-10, 11, .2), MSE_const_25, MSE_const_75, alpha = 0.5)
plt.plot(np.arange(-10, 11, .2), MSE_const_median.reshape(105), label="Constrained")

plt.fill_between(np.arange(-10, 11, .2), MSE_unconst_25, MSE_unconst_75, alpha = 0.5)
plt.plot(np.arange(-10, 11, .2), MSE_unconst_median.reshape(105), label="Linear MMSE")

plt.fill_between(np.arange(-10, 11, .2), MSE_combine_25, MSE_combine_75, alpha = 0.5)
plt.plot(np.arange(-10, 11, .2), MSE_combine_median.reshape(105), label="Combine")

plt.xlim((-10, 10))
plt.ylim((0, 10))
plt.legend(loc=2)   
plt.xlabel("z")
plt.ylabel("MSE(z)")