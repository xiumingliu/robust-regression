# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:35:17 2019

@author: Administrator
"""

from sklearn import preprocessing
from sklearn.datasets import load_boston
import numpy as np
from numpy.linalg import inv
import seaborn as sns
from cvxopt.solvers import qp
from cvxopt.solvers import cp
from cvxopt import matrix

# Data
boston = load_boston()
print(boston.DESCR)
data = np.vstack((np.transpose(boston.data), boston.target))

# Training and testing data
size_train = 400
size_test = 106

data_train = data[:, 0:size_train]
data_test = data[:, size_train:]

# Preprocessing, standerlize
scaler = preprocessing.StandardScaler().fit(np.transpose(data_train))
data_train = np.transpose(scaler.transform(np.transpose(data_train))) 
data_test = np.transpose(scaler.transform(np.transpose(data_test)))

# Split to x, y, z
# feature = (x, z), target = (y)
d = 9   # dimension of x
q = 4   # dimension of z

feature_train = data_train[0:13, :]
feature_x_train = feature_train[0:d, :]
feature_z_train = feature_train[d:, :]
target_train = data_train[13:, :]

feature_test = data_test[0:13, :]
feature_x_test = feature_test[0:d, :]
feature_z_test = feature_test[d:, :]
target_test = data_test[13:, :]

# Covariance matrices
cov_data_train = np.cov(data_train)
#ax = sns.heatmap(cov_data_train, vmin=-1, vmax=1)

alpha_beta_joint = np.dot(cov_data_train[13, 0:13].reshape(1,13), inv(cov_data_train[0:13, 0:13]))
alpha = np.transpose(alpha_beta_joint[:, 0:d])
beta = np.transpose(alpha_beta_joint[:, d:])

Gamma = np.transpose(np.dot(cov_data_train[0:d, d:d+q], inv(cov_data_train[d:d+q, d:d+q])))

Cov_xx = cov_data_train[0:d, 0:d]
Cov_xy = cov_data_train[0:d, 13].reshape(d, 1)

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

w_const = qp(Q, p, A, b)['x']

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

