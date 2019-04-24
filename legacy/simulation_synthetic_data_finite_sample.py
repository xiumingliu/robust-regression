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
import matplotlib.gridspec as gridspec
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
#MSE_lowerbound = np.zeros((105, 1))
MSE_const = np.zeros((105, iter_max))
MSE_unconst = np.zeros((105, iter_max))
MSE_biweight = np.zeros((105, iter_max))
MSE_combine = np.zeros((105, iter_max))
#MSE_2stages = np.zeros((105, 1))
for iter in range(iter_max):
    
    t = time.time()
    
    # Generated finite sample (size N) training data
    N = 1000
    xzy = np.random.multivariate_normal(np.zeros(((d+q+1))), Cov_xzy, N)
    
    XZ = xzy[:, 0:d+q]
    X = xzy[:, 0:d]
    Z = xzy[:, d:d+q].reshape(N)
    Y = xzy[:, -1]
    D_j = np.zeros(d)
    for j in range (d):
        D_j[j] = np.sqrt(np.linalg.norm(xzy[:, j])**2/N)
    D = np.diag(D_j)
    
    # =============================================================================
    # Training (Data fitting OLS-SPICE)
    # =============================================================================
    
    w = cp.Variable(d)
    
    def loss_fn(X, Y, w):
        return cp.norm(cp.matmul(X, w) - Y) 
    
    def regularizer(w):
        return cp.norm(cp.matmul(D, w), 1)
    
#    def objective_fn(X, Y, w):
#        return loss_fn(X, Y, w) + regularizer(w)
    
    def objective_fn(X, Y, w):
        return loss_fn(X, Y, w)
        
    constraints = [cp.sum(cp.matmul(np.transpose(Z),cp.matmul(X, w) - Y)) == 0]
    
    problem = cp.Problem(cp.Minimize(objective_fn(X, Y, w)), constraints)
    problem.solve()
    
    con_value = np.dot(np.transpose(Z), np.dot(X, w.value) - Y)
    
    w_uncon = cp.Variable(d)
    problem2 = cp.Problem(cp.Minimize(objective_fn(X, Y, w_uncon)))
    problem2.solve()
    
    # =============================================================================
    # Predict z first, then y  
    # =============================================================================
    
    w_z = cp.Variable(d)
    
    def loss_fn_2(X, Z, w_z):
        return cp.norm(cp.matmul(X, w_z) - Z) 
    
#    def objective_fn_2(X, Z, w_z):
#        return loss_fn_2(X, Z, w_z) + regularizer(w_z)
        
    def objective_fn_2(X, Z, w_z):
        return loss_fn_2(X, Z, w_z)
    
    problem3 = cp.Problem(cp.Minimize(objective_fn_2(X, Z, w_z)))
    problem3.solve()
    
    # =============================================================================
    # Testing
    # =============================================================================
    
    w_const = w.value[0:d]
    w_unconst = w_uncon.value[0:d]
    Theta = w_z.value
    
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
    num_test = 100
    for j in range(0, 105, 1):
        z = -10 + j*.2
        mu_xy_z, Cov_xy_z = conditional_mean(z, 0, 0, Cov_xyz)
        y_true = np.zeros((num_test,1))
    #    y_predict_lowerbound = np.zeros((num_test,1))
        y_predict_const = np.zeros((num_test,1))
        y_predict_unconst = np.zeros((num_test,1))
        z_predict_2stage = np.zeros((num_test,1))
    #    y_predict_2stage = np.zeros((num_test,1))
        y_predict_biweight = np.zeros((num_test,1))
        y_predict_combine = np.zeros((num_test,1))
        
        for i in range(0, num_test):
            xy = np.random.multivariate_normal(mu_xy_z.reshape(d+1), Cov_xy_z)
            
            y_true[i] = xy[d:]
            
            # Lower bound 
    #        y_predict_lowerbound[i] = np.dot(alpha_beta_joint, np.vstack((xy[0:d].reshape(d,1), np.array(z).reshape(q,1))).reshape(d+q,1))
            
            # Robust 
            y_predict_const[i] = np.dot(np.transpose(w_const), xy[0:d].reshape(d,1))
            
            # LMMSE
            y_predict_unconst[i] = np.dot(np.transpose(w_unconst), xy[0:d].reshape(d,1))
            
            z_predict_2stage[i] = np.dot(np.transpose(Theta), xy[0:d].reshape(d,1))
    #        y_predict_2stage[i] = np.dot(alpha_beta_joint, np.vstack((xy[0:d].reshape(d,1), z_predict_2stage[i].reshape(q,1))).reshape(d+q,1))
            
            # Biweight
            maha_distance = distance.mahalanobis(z_predict_2stage[i], 0, 1)
            if maha_distance > T:
                y_predict_biweight[i] = y_predict_const[i]
            else:
                y_predict_biweight[i] = y_predict_unconst[i]    
                
            # Combine
            a = 1/(1+np.exp(-K*(maha_distance-T)))
            b = 1 - a
            y_predict_combine[i] = a*y_predict_const[i] + b*y_predict_unconst[i]
            
        # MSE
    #    MSE_lowerbound[j, iter] = mean_squared_error(y_true, y_predict_lowerbound)
        MSE_const[j, iter] = mean_squared_error(y_true, y_predict_const) 
        MSE_unconst[j, iter] = mean_squared_error(y_true, y_predict_unconst) 
    #    MSE_2stages[j, iter] = mean_squared_error(y_true, y_predict_2stage) 
        MSE_biweight[j, iter] = mean_squared_error(y_true, y_predict_biweight) 
        MSE_combine[j, iter] = mean_squared_error(y_true, y_predict_combine) 
        
    # do stuff
    elapsed = time.time() - t
    print(iter)
    print(elapsed)
        
MSE_const_average = np.mean(MSE_const, axis=1)
MSE_unconst_average = np.mean(MSE_unconst, axis=1)
MSE_biweight_average = np.mean(MSE_biweight, axis=1)
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
    
#plt.figure(figsize=(5, 5))
#gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3])
#
#ax0 = plt.subplot(gs[0])
##sns.lineplot(np.arange(-10, 11, .2), MSE_lowerbound.reshape(105), color='k', label="Lower bound")
#sns.lineplot(np.arange(-10, 11, .2), MSE_const_average.reshape(105), label="Constrained")
#sns.lineplot(np.arange(-10, 11, .2), MSE_unconst_average.reshape(105), label="Linear MMSE") 
#plt.setp(ax0.get_xticklabels(), visible=False)
##sns.regplot(np.arange(-10, 11, .2), MSE_2stages.reshape(105), order=3, label="Two-stage") 
##plt.plot(z, rz.pdf(z), 'k-', lw=2, label='p(z)')
#plt.xlim((-10, 10))
#plt.ylim((0, 10))
#plt.legend(loc=2)   
##plt.xlabel("z")
#plt.ylabel("MSE(z)")
#
#ax1 = plt.subplot(gs[1])
##plt.figure(figsize=(5, 5))
#plt.plot(z, rz.pdf(z), 'k-', lw=2, label='p(z)')
#plt.plot(z_x, rz_x.pdf(z_x).reshape(100), 'k:', lw=2, label='p(z | x = [3, 3])')
#plt.xlim((-10, 10))
#plt.ylim((0, 1))
#plt.legend(loc=2)   
#plt.xlabel("z")
#plt.ylabel("PDF")

MSE_const_var = np.std(MSE_const, axis=1)
MSE_unconst_var = np.std(MSE_unconst, axis=1)
MSE_biweight_var = np.std(MSE_biweight, axis=1)
MSE_combine_var = np.std(MSE_combine, axis=1)

MSE_const_median = np.median(MSE_const, axis=1)
MSE_unconst_median = np.median(MSE_unconst, axis=1)
MSE_biweight_median = np.median(MSE_biweight, axis=1)
MSE_combine_median = np.median(MSE_combine, axis=1)

MSE_const_75 = np.percentile(MSE_const, 75, axis=1)
MSE_unconst_75 = np.percentile(MSE_unconst, 75, axis=1)
MSE_biweight_75 = np.percentile(MSE_biweight, 75, axis=1)
MSE_combine_75 = np.percentile(MSE_combine, 75, axis=1)

MSE_const_25 = np.percentile(MSE_const, 25, axis=1)
MSE_unconst_25 = np.percentile(MSE_unconst, 25, axis=1)
MSE_biweight_25 = np.percentile(MSE_biweight, 25, axis=1)
MSE_combine_25 = np.percentile(MSE_combine, 25, axis=1)

#plt.figure(figsize=(5, 3.5))
#plt.fill_between(np.arange(-10, 11, .2), MSE_const_average.reshape(105)-MSE_const_var, MSE_const_average.reshape(105)+MSE_const_var, alpha = 0.5)
#plt.plot(np.arange(-10, 11, .2), MSE_const_average.reshape(105), label="Constrained")
#
#plt.fill_between(np.arange(-10, 11, .2), MSE_unconst_average.reshape(105)-MSE_unconst_var, MSE_unconst_average.reshape(105)+MSE_unconst_var, alpha = 0.5)
#plt.plot(np.arange(-10, 11, .2), MSE_unconst_average.reshape(105), label="Linear MMSE")
#
##plt.fill_between(np.arange(-10, 11, .2), MSE_biweight_average.reshape(105)-MSE_biweight_var, MSE_biweight_average.reshape(105)+MSE_biweight_var, alpha = 0.5)
##plt.plot(np.arange(-10, 11, .2), MSE_biweight_average.reshape(105), label="Binary Weight")
#
#plt.fill_between(np.arange(-10, 11, .2), MSE_combine_average.reshape(105)-MSE_combine_var, MSE_combine_average.reshape(105)+MSE_combine_var, alpha = 0.5)
#plt.plot(np.arange(-10, 11, .2), MSE_combine_average.reshape(105), label="Combine")
#
#plt.xlim((-10, 10))
#plt.ylim((0, 10))
#plt.legend(loc=2)   
#plt.xlabel("z")
#plt.ylabel("MSE(z)")


plt.figure(figsize=(5, 3.5))
plt.fill_between(np.arange(-10, 11, .2), MSE_const_25, MSE_const_75, alpha = 0.5)
plt.plot(np.arange(-10, 11, .2), MSE_const_median.reshape(105), label="Constrained")

plt.fill_between(np.arange(-10, 11, .2), MSE_unconst_25, MSE_unconst_75, alpha = 0.5)
plt.plot(np.arange(-10, 11, .2), MSE_unconst_median.reshape(105), label="Linear MMSE")

#plt.fill_between(np.arange(-10, 11, .2), MSE_biweight_average.reshape(105)-MSE_biweight_var, MSE_biweight_average.reshape(105)+MSE_biweight_var, alpha = 0.5)
#plt.plot(np.arange(-10, 11, .2), MSE_biweight_average.reshape(105), label="Binary Weight")

plt.fill_between(np.arange(-10, 11, .2), MSE_combine_25, MSE_combine_75, alpha = 0.5)
plt.plot(np.arange(-10, 11, .2), MSE_combine_median.reshape(105), label="Combine")

plt.xlim((-10, 10))
plt.ylim((0, 10))
plt.legend(loc=2)   
plt.xlabel("z")
plt.ylabel("MSE(z)")


#plt.figure(figsize=(5, 3.5))
##sns.lineplot(np.arange(-10, 11, .2), MSE_lowerbound.reshape(105), color='k', label="Lower bound")
#sns.lineplot(np.arange(-10, 11, .2), MSE_const_average.reshape(105), label="Constrained")
#sns.lineplot(np.arange(-10, 11, .2), MSE_unconst_average.reshape(105), label="Linear MMSE") 
#sns.lineplot(np.arange(-10, 11, .2), MSE_biweight_average.reshape(105), label="Binary Weight") 
#sns.lineplot(np.arange(-10, 11, .2), MSE_combine_average.reshape(105), label="Combine") 
#plt.xlim((-10, 10))
#plt.ylim((0, 10))
#plt.legend(loc=2)   
#plt.xlabel("z")
#plt.ylabel("MSE(z)")




