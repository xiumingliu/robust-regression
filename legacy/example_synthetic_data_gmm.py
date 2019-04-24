# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:35:17 2019

@author: Administrator
"""

from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
from scipy.spatial import distance
#from scipy.special import expit
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cvxopt.solvers import qp
from cvxopt import matrix

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

# =============================================================================
# Data
# =============================================================================

# Split to x, y, z
# feature = (x, z), target = (y)
d = 2    # dimension of x
q = 1    # dimension of z

# =============================================================================
# Training
# =============================================================================

## Covariance matrices
Cov_xzy = np.array([[1, .25, .25, .5],
                    [.25, 1, .25, .5],
                    [.25, .25, 1, .5],
                    [.5, .5, .5, 1]])
#LA.eig(Cov_xzy)
#plt.figure(figsize=(6, 5))
#sns.heatmap(Cov_xzy)

# Mixture Gaussian: p(x, z, y) ~ pi_k * N(mu_k, Sigma_k)
Cat = 3

pi = np.array([.1, .8, .1])
# z concentrates at three different locations: -3, 0, 3
mu = np.array([[0, 0, 0], 
               [0, 0, 0],
               [-5, 0, 5],
               [-1, 0, 1]])
# Same Sigma for all categories
Sigma = np.array([[1, .25, .25, .5],
                  [.25, 1, .25, .5],
                  [.25, .25, 1, .5],
                  [.5, .5, .5, 1]])

alpha_beta_joint = np.dot(Cov_xzy[-1, 0:-1].reshape(1,(d+q)), inv(Cov_xzy[0:-1, 0:-1]))
alpha = np.transpose(alpha_beta_joint[:, 0:d])
beta = np.transpose(alpha_beta_joint[:, d:])

Gamma = np.transpose(np.dot(Cov_xzy[0:d, d:d+q], inv(Cov_xzy[d:d+q, d:d+q])))

Cov_xx = Cov_xzy[0:d, 0:d]
Cov_xy = Cov_xzy[0:d, -1].reshape(d, 1)

# =============================================================================
# A linear equality constrained convex quadratic optimization problem 
# https://cvxopt.org/userguide/coneprog.html#quadratic-programming
# min w'Qw + p'w
# s.t Aw = b
# =============================================================================

Q = 2*matrix(Cov_xx)
p = -2*matrix(Cov_xy)
A = matrix(Gamma)
b = matrix(np.dot(Gamma,alpha) + beta)

G = matrix(0.0, (d,d))
h = matrix(0.0, (d,1))

w_const = qp(Q, p, G, h, A, b)['x']
w_const = np.array(w_const)

# =============================================================================
# An unconstrained convex quadratic optimization problem 
# https://cvxopt.org/userguide/coneprog.html#quadratic-programming
# min w'Qw + p'w
# =============================================================================

w_unconst = np.dot(inv(Cov_xx), Cov_xy)

# =============================================================================
# Predict z first, then y  
# =============================================================================

Theta = np.transpose(np.dot(Cov_xzy[d:d+q, 0:d], inv(Cov_xzy[0:d, 0:d])))

# =============================================================================
# Testing
# =============================================================================

# Threshold 
T = .5
K = 10

# Generate testing data
# (x, y, z) ~ N(mu, Cov)
Cov_xyz = np.array([[1, .25, .5, .25,],
                    [.25, 1, .5, .25,],
                    [.5, .5, 1, .5],
                    [.25, .25, .5, 1]])

mu_xyz = np.array([[0, 0, 0], 
               [0, 0, 0],
               [-1, 0, 1],
               [-5, 0, 5]])

# (x, y | z) ~ N(mu_xy_z, Cov_xy_z)
def conditional_mean(z, mu_z, mu_xy, Cov_xyz):
    mu_xy_z = mu_xy + np.dot(np.dot(Cov_xyz[0:d+1, d+1:], inv(Cov_xyz[d+1:, d+1:])), (z-mu_z))  
    Cov_xy_z = Cov_xyz[0:d+1, 0:d+1] - np.dot(np.dot(Cov_xyz[0:d+1, d+1:], inv(Cov_xyz[d+1:, d+1:])), np.transpose(Cov_xyz[0:d+1, d+1:]))
    return mu_xy_z, Cov_xy_z

MSE_const = np.zeros((105, 1))
MSE_unconst = np.zeros((105, 1))
MSE_combine = np.zeros((105, 1))

N = 5000


for j in range(0, 105, 1):
    z = -10 + j*.2
#        mu_xy_z, Cov_xy_z = conditional_mean(z, 0, 0, Cov_xyz)
    
    
    pi_xy_z = np.zeros(Cat)
    # z concentrates at three different locations: -3, 0, 3
    mu_xy_z = np.zeros((d+1, Cat))
    # Same Sigma for all categories
    
    pdf_z = np.zeros(Cat)
    for k in range(Cat):
        pdf_z[k] = norm.pdf(z, mu[d, k], 1)*pi[k]
    pdf_z = np.sum(pdf_z)    
    
    
    for k in range(Cat):
        this_mu_new, Cov_xy_z = conditional_mean(z, mu[d, k], mu_xyz[0:d+1, k].reshape((3, 1)), Cov_xyz)
        mu_xy_z[:, k] = this_mu_new.reshape((d+1))
        pi_xy_z[k] = norm.pdf(z, mu[d, k], 1)*pi[k]/pdf_z
    
    y_true = np.zeros((N,1))
    y_predict_const = np.zeros((N,1))
    y_predict_unconst = np.zeros((N,1))
    z_predict_2stage = np.zeros((N,1))
    y_predict_combine = np.zeros((N,1))
    
    for i in range(0, N):
#            xy = np.random.multivariate_normal(mu_xy_z.reshape(d+1), Cov_xy_z)
        
        k = np.nonzero(np.random.multinomial(1, pi_xy_z))
        xy = np.random.multivariate_normal(mu_xy_z[:, k].reshape(d+1), Cov_xy_z)
        
        y_true[i] = xy[d:]
                    
        # Robust 
        y_predict_const[i] = np.dot(np.transpose(w_const), xy[0:d].reshape(d,1))
        
        # LMMSE
        y_predict_unconst[i] = np.dot(np.transpose(w_unconst), xy[0:d].reshape(d,1))
        
        z_predict_2stage[i] = np.dot(np.transpose(Theta), xy[0:d].reshape(d,1))
        
        # Biweight
        maha_distance = distance.mahalanobis(z_predict_2stage[i], 0, 1)
            
        # Combine
        a = 1/(1+np.exp(-K*(maha_distance-T)))
        b = 1 - a
        y_predict_combine[i] = a*y_predict_const[i] + b*y_predict_unconst[i]
        
    # MSE
    MSE_const[j] = mean_squared_error(y_true, y_predict_const) 
    MSE_unconst[j] = mean_squared_error(y_true, y_predict_unconst) 
    MSE_combine[j] = mean_squared_error(y_true, y_predict_combine) 

# =============================================================================
# Visualization of results    
# =============================================================================
    
xzy = np.zeros((d+q+1, 1000))
for n in range(1000):
    k = np.nonzero(np.random.multinomial(1, pi))
    xzy[:, n] = np.random.multivariate_normal(mu[:, k].reshape(q+d+1), Sigma)    
    
#plt.figure(figsize=(5, 3.5))
#plt.plot(np.arange(-10, 11, .2), MSE_combine[:, 0].reshape(105), 'g-', label=r"$d_0 = 0.5$") 
#plt.plot(np.arange(-10, 11, .2), MSE_const[:, 0].reshape(105), 'b:', label=r"$constrained$") 
#plt.plot(np.arange(-10, 11, .2), MSE_unconst.reshape(105), 'r--', label=r"$unconstrained$") 
#plt.hist(xzy[d, :], density='True', label='Histogram', color = 'k')
#plt.xlim((-10, 10))
#plt.ylim((0, 10))
#plt.legend(loc=2)   
#plt.xlabel("$z$")
#plt.ylabel("MSE($z$)")

#plt.savefig("K_1.pdf", bbox_inches='tight')

plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3])
ax0 = plt.subplot(gs[0])

plt.plot(np.arange(-10, 11, .2), MSE_combine.reshape(105), 'g-', label=r"$a\bm{w}_{MMSE} + (1-a)\bm{w}_{C}$")

plt.plot(np.arange(-10, 11, .2), MSE_unconst.reshape(105), 'r--', label=r"$\bm{w}_{MMSE}$")

plt.plot(np.arange(-10, 11, .2), MSE_const.reshape(105), 'b:', label=r"$\bm{w}_{C}$")

plt.xlim((-10, 10))
plt.ylim((0, 10))
plt.legend(loc=2)   
#plt.xlabel("z")
plt.ylabel("MSE($z$)")

ax1 = plt.subplot(gs[1])
#plt.figure(figsize=(5, 5))
plt.hist(xzy[d, :], density='True', label='Histogram', color = 'k')
plt.xlim((-10, 10))
plt.ylim((0, 1))
plt.legend(loc=2)   
plt.xlabel("$z$")
plt.ylabel("Density")

