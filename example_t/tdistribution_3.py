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

import seaborn as sns

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
df = 100

# Covariance matrices of X
Sigma = np.array([[1, .25, .25, .25],
                  [.25, 1, .25, .25],
                  [.25, .25, 1, .25],
                  [.25, .25, .25, 1]])




    
# Generated finite sample (size N) training data
N = 1000
#    X = np.transpose(multivariate_t_rvs(np.zeros(((d+q))), Sigma, df=df, n = N))
X = np.transpose(np.random.multivariate_normal(np.zeros(((d))), Sigma, N))
#    xz = np.transpose(multivariatet(np.zeros(((d+q))), Sigma, df, N))

#    XZ = xz[0:d+q, :]
#    X = xz[0:d, :].reshape(d, N)
#    Z = xz[d:d+q, :].reshape(q, N)



Z = np.dot(np.array([.5, .5, .5, .5]), X) + np.random.standard_t(df, size=N)
#    Z = np.dot(np.array([.5, .5, .5, .5]), X) + np.random.normal(0, 1, size = N)
Z = Z.reshape((q, N))

Y = np.dot(np.array([.5, .5, .5, .5]), X) + .5*Z

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

#def regularizer_1(theta):
#    return cp.norm(cp.matmul(D_new, theta), 1)

def objective_fn_1(A, B, theta):
    return loss_fn_1(A, B, theta) + regularizer_1(theta)

def objective_fn_1(A, B, theta):
    return loss_fn_1(A, B, theta)

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
K = 100

N_test = 1000000
X_test = np.transpose(np.random.multivariate_normal(np.zeros(((d))), Sigma, N_test))

Z_test = np.dot(np.array([.5, .5, .5, .5]), X_test) + np.random.standard_t(df, size=N_test)
#    Z_test = np.dot(np.array([.5, .5, .5, .5]), X_test) + np.random.normal(0, 1, size = N_test)
Z_test = Z_test.reshape((q, N_test))

Y_test = np.dot(np.array([.5, .5, .5, .5]), X_test) + .5*Z_test
    
Y_predict_const = np.zeros((N_test,1))
Y_predict_unconst = np.zeros((N_test,1))
Z_predict = np.zeros((N_test,1))
Y_predict_combine = np.zeros((N_test,1))    

# Robust 
Y_predict_const = np.dot(np.transpose(w_c), X_test)

# LMMSE
Y_predict_unconst = np.dot(np.transpose(w_opt), X_test)

#    Z_predict = np.dot(np.transpose(w_z), X_test)
#        
#    # Combine
#    maha_distance = distance.mahalanobis(z_predict[i], 0, 1)
#    a = 1/(1+np.exp(-K*(maha_distance-T)))
#    b = 1 - a
#    y_predict_combine[i] = a*y_predict_const[i] + b*y_predict_unconst[i]
        
# Residual
Residual_const = Y_test - Y_predict_const
Residual_unconst = Y_test - Y_predict_unconst
Residual_const_abs =  np.abs(Residual_const)
Residual_unconst_abs =  np.abs(Residual_unconst)
# =============================================================================
# Visualization of results    
# =============================================================================

plt.figure(figsize=(5, 5))
sns.regplot(Z_test.reshape(N_test), np.abs(Residual_const).reshape(N_test), x_bins=30, order=3, fit_reg=0, label='$\bm{w}_{C}$')
sns.regplot(Z_test.reshape(N_test), np.abs(Residual_unconst).reshape(N_test), x_bins=30, order=3, fit_reg=0, label='$\bm{w}_{MMSE}$')
plt.xlabel("$z \sim St(0, 1, v), v = 100$")
plt.ylabel("Absolute residual")
