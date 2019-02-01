# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:35:17 2019

@author: Administrator
"""

from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy.linalg import inv
import seaborn as sns
from cvxopt.solvers import qp
from cvxopt import matrix

# =============================================================================
# Data
# =============================================================================

# Data
boston = load_boston()
print(boston.DESCR)
data = np.vstack((np.transpose(boston.data), boston.target))

# Training and testing data
size_train = 406
size_test = 100

data_train = data[:, 0:size_train]
data_test = data[:, size_train:]

## selected features (5, 9, 10, 12)
index = (9, 10, 12, 13)
data_train = data[index, 0:size_train]
data_test = data[index, size_train:]

# Preprocessing, standerlize
scaler = preprocessing.StandardScaler().fit(np.transpose(data_train))
data_train = np.transpose(scaler.transform(np.transpose(data_train))) 
data_test = np.transpose(scaler.transform(np.transpose(data_test)))

# Split to x, y, z
# feature = (x, z), target = (y)
d = 2    # dimension of x
q = 1    # dimension of z

feature_train = data_train[0:-1, :]
feature_x_train = feature_train[0:d, :]
feature_z_train = feature_train[d:, :]
target_train = data_train[-1:, :]

feature_test = data_test[0:-1, :]
feature_x_test = feature_test[0:d, :]
feature_z_test = feature_test[d:, :]
target_test = data_test[-1:, :]

# =============================================================================
# Training
# =============================================================================

# Covariance matrices
cov_data_train = np.cov(data_train)
ax = sns.heatmap(cov_data_train)

alpha_beta_joint = np.dot(cov_data_train[-1, 0:-1].reshape(1,(d+q)), inv(cov_data_train[0:-1, 0:-1]))
alpha = np.transpose(alpha_beta_joint[:, 0:d])
beta = np.transpose(alpha_beta_joint[:, d:])

Gamma = np.transpose(np.dot(cov_data_train[0:d, d:d+q], inv(cov_data_train[d:d+q, d:d+q])))

Cov_xx = cov_data_train[0:d, 0:d]
Cov_xy = cov_data_train[0:d, -1].reshape(d, 1)

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

# Average MSE
target_unconst = np.dot(np.transpose(w_unconst), feature_x_test)
MSE_unconst = mean_squared_error(target_test, target_unconst)

target_const = np.dot(np.transpose(w_const), feature_x_test)
MSE_const = mean_squared_error(target_test, target_const)

# For specific missing features
# Visualize the impact of missing feature "LSTAT"
sns.jointplot(feature_z_train[-1, :].reshape(size_train,1), target_train.reshape(size_train,1), kind="reg").set_axis_labels("LSTAT", "MEDV")

error_const = (target_const-target_test)
error_unconst = (target_unconst-target_test)

sns.jointplot(feature_z_test[-1, :].reshape(size_test,1), error_const.reshape(size_test,1), kind="reg").set_axis_labels("LSTAT", "MEDV prediction error")
sns.jointplot(feature_z_test[-1, :].reshape(size_test,1), error_unconst.reshape(size_test,1), kind="reg").set_axis_labels("LSTAT", "MEDV prediction error")
