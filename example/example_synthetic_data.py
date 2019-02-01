# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:35:17 2019

@author: Administrator
"""

from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
import numpy as np
from numpy.linalg import inv
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt.solvers import qp
from cvxopt import matrix

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

# Covariance matrices
Cov_xzy = np.array([[1, 0, .3, .5],
                    [0, 1, .3, .6],
                    [.3, .3, 1, .7],
                    [.5, .6, .7, 1]])
plt.figure()
sns.heatmap(Cov_xzy)

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

Q = matrix(Cov_xx)
p = matrix(-2*Cov_xy)
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

Q = matrix(Cov_xx)
p = matrix(-2*Cov_xy)

w_unconst = qp(Q, p)['x']
w_unconst = np.array(w_unconst)

# =============================================================================
# Testing
# =============================================================================

# Generate testing data
# (x, y, z) ~ N(mu, Cov)
Cov_xyz = np.array([[1, 0, .5, .3,],
                    [0, 1, .6, .3,],
                    [.5, .6, 1, .7],
                    [.3, .3, .7, 1]])
plt.figure()
sns.heatmap(Cov_xyz)

# (x, y | z) ~ N(mu_xy_z, Cov_xy_z)
def conditional_mean(z, mu_z, mu_xy, Cov_xyz):
    mu_xy_z = mu_xy + np.dot(np.dot(Cov_xyz[0:d+1, d+1:], inv(Cov_xyz[d+1:, d+1:])), (z-mu_z))  
    Cov_xy_z = Cov_xyz[0:d+1, 0:d+1] - np.dot(np.dot(Cov_xyz[0:d+1, d+1:], inv(Cov_xyz[d+1:, d+1:])), np.transpose(Cov_xyz[0:d+1, d+1:]))
    return mu_xy_z, Cov_xy_z

#mu_xy_z, Cov_xy_z = conditional_mean(1, 0, 0, Cov_xyz)

for z in range(-5, 5, 1):
    mu_xy_z, Cov_xy_z = conditional_mean(z, 0, 0, Cov_xyz)
    y_true = np.zeros((100,1))
    y_predict_const = np.zeros((100,1))
    y_predict_unconst = np.zeros((100,1))
    for i in range(0, 100):
        xy = np.random.multivariate_normal(mu_xy_z.reshape(d+1), Cov_xy_z)
        y_true[i] = xy[d:]
        y_predict_const[i] = np.dot(np.transpose(w_const), xy[0:d].reshape(d,1))
        y_predict_unconst[i] = np.dot(np.transpose(w_unconst), xy[0:d].reshape(d,1))
        
