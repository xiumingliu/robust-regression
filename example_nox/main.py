# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:45:47 2019

@author: Administrator
"""

import scipy.io as sio
from scipy.spatial import distance
import datetime
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

data = sio.loadmat('data')
time_begin = datetime.datetime(2006, 1, 1)
time_end = datetime.datetime(2015, 12, 31)
time = np.arange(time_begin, time_end+datetime.timedelta(days=1), datetime.timedelta(days=1)).astype('datetime64[D]')

nox = data['no2']
nox[np.isnan(nox)] = 0
temperature = data['temperature']
temperature[np.isnan(temperature)] = 0
#rh = data['rh']
#rh[np.isnan(rh)] = 0
ozone = data['o3']
ozone[np.isnan(ozone)] = 0

nox_daily = nox.reshape(3652, 24)
nox_daily_average = np.average(nox_daily, axis=1)
pt = PowerTransformer()
pt.fit(nox_daily_average.reshape(-1, 1))
nox_daily_average_transformed = pt.transform(nox_daily_average.reshape(-1, 1)).reshape(-1)

#plt.figure()
#plt.hist(nox_daily_average_transformed)

#temperature_daily = temperature.reshape(3652, 24)
#temperature_daily_average = np.average(temperature_daily, axis=1)
#pt_temperature = PowerTransformer()
#pt_temperature.fit(temperature_daily_average.reshape(-1, 1))
#temperature_daily_average_transformed = pt_temperature.transform(temperature_daily_average.reshape(-1, 1)).reshape(-1)
#
#plt.figure()
#plt.hist(temperature_daily_average_transformed)

#rh_daily = rh.reshape(3652, 24)
#rh_daily_average = np.average(rh_daily, axis=1)

ozone_daily = ozone.reshape(3652, 24)
ozone_daily_average = np.average(ozone_daily, axis=1)
pt_ozone = PowerTransformer()
pt_ozone.fit(ozone_daily_average.reshape(-1, 1))
ozone_daily_average_transformed = pt_ozone.transform(ozone_daily_average.reshape(-1, 1)).reshape(-1)

#plt.figure()
#plt.hist(ozone_daily_average_transformed)
#
#plt.figure()
#plt.plot(nox_daily_average_transformed[0:10000])
##plt.plot(temperature_daily_average_transformed[0:10000])
##plt.plot(rh_daily_average[0:10000])
#plt.plot(ozone_daily_average_transformed[0:10000])

def broadcasting_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]

# =============================================================================
# Data
# =============================================================================
L = 7
d = 2*L
X = np.concatenate((np.transpose(broadcasting_app(nox_daily_average_transformed, L, 1)), np.transpose(broadcasting_app(ozone_daily_average_transformed, L, 1))))
#X = np.concatenate((X, np.ones((1, np.shape(X)[1]))))
X = X[:, 0:-1]

Y = nox_daily_average_transformed[L:]
N = np.size(Y)
Y = Y.reshape(1, N)

q = 1
Z = ozone_daily_average_transformed[L:].reshape((q, N))

time = time[d:]
index_date_training_end = np.where(time == np.datetime64('2012-12-31'))[0][0]

# Training data
N_training = index_date_training_end+1
X_training = X[:, 0:index_date_training_end+1]
Y_training = Y[0, 0:index_date_training_end+1].reshape((1, N_training))
Z_training = Z[0, 0:index_date_training_end+1].reshape((q, N_training))

# Test data
N_test = np.size(Y) - N_training
X_test = X[:, index_date_training_end+1:]
Y_test = Y[0, index_date_training_end+1:].reshape((1, N_test))
Z_test = Z[0, index_date_training_end+1:].reshape((q, N_test))

# =============================================================================
# Model
# =============================================================================
D_j = np.zeros(d)
for j in range (d):
    D_j[j] = np.sqrt(np.linalg.norm(X_training[j, :])**2/N_training)
D = np.diag(D_j)

M = np.dot(Z_training, np.transpose(X_training))
M_dagger = np.linalg.pinv(M)

B = Y_training - np.dot(np.transpose(np.dot(M_dagger, np.dot(Z_training, np.transpose(Y_training)))), X_training)
A = np.dot(np.transpose(np.identity(d) - np.dot(M_dagger, M)), X_training)

D_j = np.zeros(d)
for j in range (d):
    D_j[j] = np.sqrt(np.linalg.norm(A[j, :])**2/N_training)
D_new = np.diag(D_j)

# =============================================================================
# Training (Data fitting OLS-SPICE)
# =============================================================================

theta = cp.reshape(cp.Variable(d), (d, 1))

def loss_fn_1(A, B, theta):
    return cp.norm(B - cp.matmul(cp.atoms.affine.transpose.transpose(theta), A)) 

def regularizer_1(theta):
    return cp.norm(cp.matmul(D_new, theta), 1)

#def objective_fn_1(A, B, theta):
#    return loss_fn_1(A, B, theta) + regularizer_1(theta)

def objective_fn_1(A, B, theta):
    return loss_fn_1(A, B, theta)

problem_1 = cp.Problem(cp.Minimize(objective_fn_1(A, B, theta)))
problem_1.solve()
theta = theta.value[0:d]
w_c = np.dot(M_dagger, np.dot(Z_training, np.transpose(Y_training))) + np.dot((np.identity(d) - np.dot(M_dagger, M)), theta)

# =============================================================================
# Unconstrained optimal    
# =============================================================================

w_opt = cp.reshape(cp.Variable(d), (d, 1))

def loss_fn_2(X_training, Y_training, w_opt):
    return cp.norm(Y_training - cp.matmul(cp.atoms.affine.transpose.transpose(w_opt), X_training)) 

def regularizer_2(w_opt):
    return cp.norm(cp.matmul(D, w_opt), 1)

#def objective_fn_2(X_training, Y_training, w_opt):
#    return loss_fn_2(X_training, Y_training, w_opt) + regularizer_2(w_opt)

def objective_fn_2(X_training, Y_training, w_opt):
    return loss_fn_2(X_training, Y_training, w_opt)

problem_2 = cp.Problem(cp.Minimize(objective_fn_2(X_training, Y_training, w_opt)))
problem_2.solve()
w_opt = w_opt.value[0:d]

# =============================================================================
# Predict z first, then y  
# =============================================================================

w_z = cp.reshape(cp.Variable(d), (d, 1))

def loss_fn_3(X_training, Z_training, w_z):
    return cp.norm(Z_training - cp.matmul(cp.atoms.affine.transpose.transpose(w_z), X_training)) 

#def objective_fn_3(X_training, Z_training, w_z):
#    return loss_fn_3(X_training, Z_training, w_z) + regularizer_2(w_z)

def objective_fn_3(X_training, Z_training, w_z):
    return loss_fn_3(X_training, Z_training, w_z)

problem_3 = cp.Problem(cp.Minimize(objective_fn_3(X_training, Z_training, w_z)))
problem_3.solve()
w_z = w_z.value

# =============================================================================
# Test
# =============================================================================
# Robust 
Y_predict_const = pt.inverse_transform(np.dot(np.transpose(w_c), X_test.reshape(d,N_test)).reshape(N_test, 1))
#Y_predict_const = np.dot(np.transpose(w_c), X_test.reshape(d,N_test)).reshape(N_test, 1)

# LMMSE
Y_predict_unconst = pt.inverse_transform(np.dot(np.transpose(w_opt), X_test.reshape(d,N_test)).reshape(N_test, 1))
#Y_predict_unconst = np.dot(np.transpose(w_opt), X_test.reshape(d,N_test)).reshape(N_test, 1)

alpha = 0.30

Z_tail_index = np.logical_or(Z.reshape(N) < -np.sqrt(1/alpha)*np.std(Z), Z.reshape(N) > np.sqrt(1/alpha)*np.std(Z))
Z_tail = np.zeros(N)
Z_tail[Z_tail_index] = 1    
Z_predict_train = np.dot(np.transpose(w_z), X.reshape(d,N))
Z_distance = np.zeros(N)
for i in range(N):
    Z_distance[i] = distance.mahalanobis(Z_predict_train[:, i], np.zeros(q), np.cov(Z_predict_train)) - np.sqrt(np.cov(Z_predict_train)/alpha)
#    Z_distance[i] = distance.mahalanobis(Z[:, i], np.zeros(q), np.cov(Z)) - np.sqrt(np.cov(Z)/alpha)
    
clf = LogisticRegression(random_state=0, solver='lbfgs', fit_intercept=False)
clf.fit(Z_distance.reshape(-1, 1), Z_tail)

# Threshold 
#T = 1
#K = 10
T = np.sqrt(np.cov(Z_predict_train)/alpha)
K = clf.coef_

Z_std = np.std(Z_training)
Z_mean = np.mean(Z_training)

Z_predict = np.dot(np.transpose(w_z), X_test.reshape(d,N_test))
# Combine
Y_predict_combine = np.zeros((N_test, 1))
for i in range(N_test):
    maha_distance = distance.mahalanobis(Z_predict[0,i], Z_mean, np.square(Z_std))
    a = 1/(1+np.exp(-K*(maha_distance-T)))
    b = 1 - a
    Y_predict_combine[i, 0] = a*Y_predict_const[i, 0] + b*Y_predict_unconst[i, 0]

# Error
Error_const = np.abs(pt.inverse_transform(Y_test.reshape(N_test).reshape(-1, 1)) - Y_predict_const.reshape(N_test, 1))
Error_unconst = np.abs(pt.inverse_transform(Y_test.reshape(N_test).reshape(-1, 1)) - Y_predict_unconst.reshape(N_test, 1))
Error_combine = np.abs(pt.inverse_transform(Y_test.reshape(N_test).reshape(-1, 1)) - Y_predict_combine.reshape(N_test, 1))

#Error_const = np.abs(Y_test.reshape(N_test).reshape(-1, 1) - Y_predict_const.reshape(N_test, 1))
#Error_unconst = np.abs(Y_test.reshape(N_test).reshape(-1, 1) - Y_predict_unconst.reshape(N_test, 1))

#plt.figure()
#plt.plot(Y_predict_const[300:400].reshape(100), label='Constrained')
#plt.plot(Y_predict_unconst[300:400].reshape(100), label='LMMSE')
#plt.plot(Y_test[0, 300:400].reshape(100, 1), label='True')
#plt.legend()

Z_test_inv_transformed = pt_ozone.inverse_transform(Z_test.reshape(N_test, 1))

#plt.figure()
#plt.scatter(Z_test.reshape(N_test, 1), np.abs(Error_const), label='Constrained')
#plt.scatter(Z_test.reshape(N_test, 1), np.abs(Error_unconst), label='LMMSE')
#plt.scatter(Z_test.reshape(N_test, 1), np.abs(Error_combine), label='Combined')
#plt.xlabel('Ozone')
#plt.ylabel('Absolute residual (NO2)')

k = 2

RMSE_const_typical = np.sqrt(np.mean(np.square(Error_const[np.logical_and(Z_test>Z_mean-k*Z_std, Z_test<Z_mean+k*Z_std)[0]])))
RMSE_unconst_typical = np.sqrt(np.mean(np.square(Error_unconst[np.logical_and(Z_test>Z_mean-k*Z_std, Z_test<Z_mean+k*Z_std)[0]])))
RMSE_combine_typical = np.sqrt(np.mean(np.square(Error_combine[np.logical_and(Z_test>Z_mean-k*Z_std, Z_test<Z_mean+k*Z_std)[0]])))

RMSE_const_tail = np.sqrt(np.mean(np.square(Error_const[np.logical_or(Z_test<Z_mean-k*Z_std, Z_test>Z_mean+k*Z_std)[0]])))
RMSE_unconst_tail = np.sqrt(np.mean(np.square(Error_unconst[np.logical_or(Z_test<Z_mean-k*Z_std, Z_test>Z_mean+k*Z_std)[0]])))
RMSE_combine_tail = np.sqrt(np.mean(np.square(Error_combine[np.logical_or(Z_test<Z_mean-k*Z_std, Z_test>Z_mean+k*Z_std)[0]])))

#RMSE_const_typical = np.sqrt(np.mean(np.square(Error_const[np.logical_and(Z_test>Z_mean-T, Z_test<Z_mean+T)[0]])))
#RMSE_unconst_typical = np.sqrt(np.mean(np.square(Error_unconst[np.logical_and(Z_test>Z_mean-T, Z_test<Z_mean+T)[0]])))
#RMSE_combine_typical = np.sqrt(np.mean(np.square(Error_combine[np.logical_and(Z_test>Z_mean-T, Z_test<Z_mean+T)[0]])))
#
#RMSE_const_tail = np.sqrt(np.mean(np.square(Error_const[np.logical_or(Z_test<Z_mean-T, Z_test>Z_mean+T)[0]])))
#RMSE_unconst_tail = np.sqrt(np.mean(np.square(Error_unconst[np.logical_or(Z_test<Z_mean-T, Z_test>Z_mean+T)[0]])))
#RMSE_combine_tail = np.sqrt(np.mean(np.square(Error_combine[np.logical_or(Z_test<Z_mean-T, Z_test>Z_mean+T)[0]])))

ratio_normal_const = RMSE_const_typical/RMSE_unconst_typical - 1
ratio_normal_combine = RMSE_combine_typical/RMSE_unconst_typical - 1

ratio_tail_const = RMSE_const_tail/RMSE_unconst_tail - 1
ratio_tail_combine = RMSE_combine_tail/RMSE_unconst_tail - 1

print("const_typical = %.4f" %ratio_normal_const)
print("const_tail = %.4f" %ratio_tail_const)
print("combine_typical = %.4f" %ratio_normal_combine)
print("combine_tail = %.4f" %ratio_tail_combine)



#plt.figure()
#plt.hist(Z_training.reshape(N_training), density=True)
#
#Z_tail_index = np.logical_or(Z_training.reshape(N_training) < Z_mean-k*Z_std, Z_training.reshape(N_training) > Z_mean+k*Z_std)
#Z_tail = np.zeros(N_training)
#Z_tail[Z_tail_index] = 1
#
#SigmaX = np.cov(X_training)
#muX = np.mean(X_training, axis = 1)
#X_distance = np.zeros(N_training)
#for i in range(N_training):
#    X_distance[i] = distance.mahalanobis(X_training[:, i], muX, SigmaX)
#
#plt.figure()
#sns.regplot(X_distance, Z_tail, logistic=True)
#plt.yticks([0, 1], ['Typical', 'Tail'])
#plt.xlabel('Mahalanobis distance between a realization of X and p(x)')