# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:35:17 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.spatial import distance
from scipy.stats import norm
from scipy.stats import t as studentt
from sklearn.metrics import mean_squared_error
import time

from math import *

import matplotlib.gridspec as gridspec
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

# =============================================================================
# written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
# =============================================================================

def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

def multivariatet(mu,Sigma,N,M):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    M = # of samples to produce
    '''
    d = len(Sigma)
    g = np.tile(np.random.gamma(N/2.,2./N,M),(d,1)).T
    Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
    return mu + Z/np.sqrt(g)

# =============================================================================
# Data
# =============================================================================

# Split to x, y, z
# feature = (x, z), target = (y)
d = 4    # dimension of x
q = 1    # dimension of z
df = 3

# Covariance matrices
Sigma = np.array([[1, .25, .25, .25, .25],
                  [.25, 1, .25, .25, .25],
                  [.25, .25, 1, .25, .25],
                  [.25, .25, .25, 1, .25], 
                  [.25, .25, .25, .25, 1]])

iter_max = 50

MSE_const = np.zeros((105, iter_max))
MSE_unconst = np.zeros((105, iter_max))
MSE_combine = np.zeros((105, iter_max))

for iter in range(iter_max):
    
    t = time.time()
    
    # Generated finite sample (size N) training data
    N = 1000
    xz = np.transpose(multivariate_t_rvs(np.zeros(((d+q))), Sigma, df=df, n = N))
#    xz = np.transpose(np.random.multivariate_normal(np.zeros(((d+q))), Sigma, N))
#    xz = np.transpose(multivariatet(np.zeros(((d+q))), Sigma, df, N))
    
    XZ = xz[0:d+q, :]
    X = xz[0:d, :].reshape(d, N)
    Z = xz[d:d+q, :].reshape(q, N)
    
    Y = np.dot(np.array([.5, .5, .5, .5]), X) + 0.5*Z
    
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
#    def conditional_mean(z, mu_z, mu_x, Cov_xz, df, q):
#        df_x_z = df + q
#        mu_x_z = mu_x + np.dot(np.dot(Cov_xz[0:d, d:], np.linalg.inv(Cov_xz[d:, d:])), (z-mu_z))  
#        Cov_x_z = Cov_xz[0:d, 0:d] - np.dot(np.dot(Cov_xz[0:d, d:], np.linalg.inv(Cov_xz[d:, d:])), np.transpose(Cov_xz[0:d, d:]))
#        alpha = (df + np.dot(np.dot(np.transpose(z - mu_z), np.linalg.inv(Cov_xz[d:, d:])), (z - mu_z)))/(df + q)
#        Cov_x_z = alpha*Cov_x_z
#        return mu_x_z, Cov_x_z, df_x_z
    
    def conditional_mean(z, mu_z, mu_x, Cov_xz):
        mu_x_z = mu_x + np.dot(np.dot(Cov_xz[0:d, d:], np.linalg.inv(Cov_xz[d:, d:])), (z-mu_z))  
        Cov_x_z = Cov_xz[0:d, 0:d] - np.dot(np.dot(Cov_xz[0:d, d:], np.linalg.inv(Cov_xz[d:, d:])), np.transpose(Cov_xz[0:d, d:]))
        return mu_x_z, Cov_x_z
    
    num_test = 100
    for j in range(0, 105, 1):
        z = -10 + j*.2
#        mu_x_z, Cov_x_z, df_x_z = conditional_mean(z, 0, 0, Sigma, df, q)
        mu_x_z, Cov_x_z = conditional_mean(z, 0, 0, Sigma)
        y_true = np.zeros((num_test,1))
        y_predict_const = np.zeros((num_test,1))
        y_predict_unconst = np.zeros((num_test,1))
        z_predict = np.zeros((num_test,1))
        y_predict_combine = np.zeros((num_test,1))
        
        for i in range(0, num_test):
#            x = multivariate_t_rvs(mu_x_z, Cov_x_z, df=df_x_z, n = 1)
            x = np.random.multivariate_normal(mu_x_z.reshape(d), Cov_x_z)
            
            y_true[i] = np.dot(np.array([.5, .5, .5, .5]), x) + 0.5*z
            
            # Robust 
            y_predict_const[i] = np.dot(np.transpose(w_c), x.reshape(d,1))
            
            # LMMSE
            y_predict_unconst[i] = np.dot(np.transpose(w_opt), x.reshape(d,1))
            
            z_predict[i] = np.dot(np.transpose(w_z), x.reshape(d,1))
                
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

rv = studentt(df=df, loc=0, scale=1)
z = np.linspace(rv.ppf(0.000001), rv.ppf(0.999999), 1000)
p_z = rv.pdf(z) 
log_p_z = np.log(p_z)


## Figure
plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])

ax0 = plt.subplot(gs[0])

plt.fill_between(np.arange(-10, 11, .2), MSE_combine_25, MSE_combine_75, alpha = 0.5, color ='g')
plt.plot(np.arange(-10, 11, .2), MSE_combine_median.reshape(105), 'g-', label=r"$a\bm{w}_{MMSE} + (1-a)\bm{w}_{C}$")

plt.fill_between(np.arange(-10, 11, .2), MSE_unconst_25, MSE_unconst_75, alpha = 0.5, color ='r')
plt.plot(np.arange(-10, 11, .2), MSE_unconst_median.reshape(105), 'r--', label=r"$\bm{w}_{MMSE}$")

plt.fill_between(np.arange(-10, 11, .2), MSE_const_25, MSE_const_75, alpha = 0.5, color ='b')
plt.plot(np.arange(-10, 11, .2), MSE_const_median.reshape(105), 'b:', label=r"$\bm{w}_{C}$")

plt.xlim((-10, 10))
plt.ylim((0, 10))
ax0.xaxis.set_visible(False)
plt.legend(loc=2)   
plt.ylabel("MSE($z$)")

ax2 = plt.subplot(gs[1])
plt.plot(z, log_p_z, label=r'$\nu = %.i$' %df, color = 'k')
#plt.hist(Z.reshape(1000), density='True', label='Histogram', color = 'k')
plt.xlim((-10, 10))
plt.ylim((-10, 0))
plt.legend(loc=2)   
plt.xlabel("$z$")
plt.ylabel("Log-scale density")

plt.savefig("v=3.pdf", bbox_inches='tight')