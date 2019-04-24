# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:35:17 2019

@author: Administrator
"""

import numpy as np
import cvxpy as cp
from scipy.spatial import distance
from scipy.stats import t as studentt
from scipy.stats import probplot
from sklearn.metrics import mean_squared_error
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}"]
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

# =============================================================================
# Data
# =============================================================================

# z ~ St(0, var, v)    
# x = z + v, where v ~ N(0, Sigma_x)
# y = x + z    


# Split to x, y, z
# feature = (x, z), target = (y)
d = 3    # dimension of x
q = 1    # dimension of z
df = 3
# Covariance matrices
Sigma_x = np.array([[1, .25, .25],
                    [.25, 1, .25],
                    [.25, .25, 1]])

iter_max = 50

MSE_const = np.zeros((105, iter_max))
MSE_unconst = np.zeros((105, iter_max))
MSE_combine = np.zeros((105, iter_max))

for iter in range(iter_max):
    
    t = time.time()
    
    # Generated finite sample (size N) training data
    N = 1000
    Z = np.transpose(np.random.standard_t(df, N)).reshape((q, N))
    
#    V = np.transpose(np.random.multivariate_normal(np.zeros(d), Sigma_x, N))
    V = np.transpose(multivariate_t_rvs(np.zeros(d), Sigma_x, df, N))
    X = np.ones((d, 1))*Z + V + 0.1*np.random.multivariate_normal(np.zeros(d), np.identity(d), N).reshape(d, N)
    
    Y = np.dot(np.array([1, 1, 1]), X) + 1*Z + 0.1*np.random.standard_normal(N)
    
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
    T = 2.5
    K = 1
    
    
    num_test = 1000
    for j in range(0, 105, 1):
        z = -10 + j*.2

        y_true = np.zeros((num_test,1))
        y_predict_const = np.zeros((num_test,1))
        y_predict_unconst = np.zeros((num_test,1))
        z_predict = np.zeros((num_test,1))
        y_predict_combine = np.zeros((num_test,1))
        
        for i in range(0, num_test):
#            x = np.random.multivariate_normal(np.ones(d)*z, Sigma_x)
            x = np.transpose(multivariate_t_rvs(np.ones(d)*z, Sigma_x, df, 1)) + 0.1*np.random.multivariate_normal(np.zeros(d), np.identity(d)).reshape(d, 1)
            
            y_true[i] = np.dot(np.array([1, 1, 1]), x) + 1*z + 0.1*np.random.standard_normal()
            
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
gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3])

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
plt.ylabel("E$[(y - \widehat{y})^2 | z]$")

ax2 = plt.subplot(gs[1])
#plt.plot(z, log_p_z, label=r'$\nu = %.i$' %df, color = 'k')
plt.plot(z, p_z, label=r'$p(z)$', color = 'k')
#plt.hist(Z.reshape(1000), density='True', label='Histogram', color = 'k')
plt.xlim((-10, 10))
plt.ylim((0.0001, 1))
plt.legend(loc=2)   
plt.xlabel("$z$")
ax2.set_yscale('log')
plt.ylabel("Density")

plt.savefig("v=3.pdf", bbox_inches='tight')

plt.figure(figsize=(5, 5))
probplot(Z.reshape(N), plot=plt)
plt.xlim((-5, 5))
plt.ylim((-20, 20))
#plt.savefig("qq.pdf", bbox_inches='tight')

#MSE_unconst_median.reshape(105)/MSE_unconst_median.reshape(105)
