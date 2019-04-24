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
from sklearn.linear_model import LogisticRegression
import time
import seaborn as sns

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
Sigma_x = np.array([[1, .5, .5],
                    [.5, 1, .5],
                    [.5, .5, 1]])

# Generated finite sample (size N) training data
N = np.int(1e3)
N_test = np.int(1e6)
max_iteration = 1

ratio_normal_const = np.zeros(max_iteration)
ratio_normal_combine = np.zeros(max_iteration)

ratio_tail_const = np.zeros(max_iteration)
ratio_tail_combine = np.zeros(max_iteration)

for iteration in range(max_iteration):
    t = time.time()
    
    Z = np.transpose(np.random.standard_t(df, N)).reshape((q, N))
    
    #    V = np.transpose(np.random.multivariate_normal(np.zeros(d), Sigma_x, N))
    V = np.transpose(multivariate_t_rvs(np.zeros(d), Sigma_x, df, N))
    X = np.ones((d, 1))*Z + V + 0.001*np.random.multivariate_normal(np.zeros(d), np.identity(d), N).reshape(d, N)
    
    Y = np.dot(np.array([1, 1, 1]), X) + 1*Z + 0.001*np.random.standard_normal(N)
    
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
    
#    def objective_fn_1(A, B, theta):
#        return loss_fn_1(A, B, theta) + regularizer_1(theta)
    
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
    
#    def objective_fn_2(X, Y, w_opt):
#        return loss_fn_2(X, Y, w_opt) + regularizer_2(w_opt)
    
    def objective_fn_2(X, Y, w_opt):
        return loss_fn_2(X, Y, w_opt)
    
    problem_2 = cp.Problem(cp.Minimize(objective_fn_2(X, Y, w_opt)))
    problem_2.solve()
    
    # =============================================================================
    # Predict z first, then y  
    # =============================================================================
    
    w_z = cp.reshape(cp.Variable(d), (d, 1))
    
    def loss_fn_3(X, Z, w_z):
        return cp.norm(Z - cp.matmul(cp.atoms.affine.transpose.transpose(w_z), X)) 
    
#    def objective_fn_3(X, Z, w_z):
#        return loss_fn_3(X, Z, w_z) + regularizer_2(w_z)
    
    def objective_fn_3(X, Z, w_z):
        return loss_fn_3(X, Z, w_z)
    
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
    
    
    # Generated finite sample (size N) testing data
    Z_test = np.transpose(np.random.standard_t(df, N_test)).reshape((q, N_test))
    
    V_test = np.transpose(multivariate_t_rvs(np.zeros(d), Sigma_x, df, N_test))
    X_test = np.ones((d, 1))*Z_test + V_test + 0.001*np.random.multivariate_normal(np.zeros(d), np.identity(d), N_test).reshape(d, N_test)
    
    Y_test = np.dot(np.array([1, 1, 1]), X_test) + 1*Z_test + 0.001*np.random.standard_normal(N_test)
    
    
    # Robust 
    Y_predict_const = np.dot(np.transpose(w_c), X_test.reshape(d,N_test))
    
    # LMMSE
    Y_predict_unconst = np.dot(np.transpose(w_opt), X_test.reshape(d,N_test))
    
    Z_predict = np.dot(np.transpose(w_z), X_test.reshape(d,N_test))
        
    # Combine
    Y_predict_combine = np.zeros((1, N_test))
    for i in range(N_test):
        maha_distance = distance.mahalanobis(Z_predict[0,i], 0, 1)
        a = 1/(1+np.exp(-K*(maha_distance-T)))
        b = 1 - a
        Y_predict_combine[0, i] = a*Y_predict_const[0, i] + b*Y_predict_unconst[0, i]
            
    # MSE
    MSE_const = mean_squared_error(Y_test, Y_predict_const) 
    MSE_unconst = mean_squared_error(Y_test, Y_predict_unconst) 
    MSE_combine = mean_squared_error(Y_test, Y_predict_combine) 
    
    Z_std = np.std(Z_test)
    
    alpha = 0.1
    
    index_tail = np.logical_or(Z_test < -np.sqrt(1/alpha)*Z_std , Z_test > np.sqrt(1/alpha)*Z_std)
    index_typical = np.logical_or(Z_test > -np.sqrt(1/alpha)*Z_std , Z_test < np.sqrt(1/alpha)*Z_std)
    
    MSE_const_tail = mean_squared_error(Y_test[index_tail], Y_predict_const[index_tail]) 
    MSE_unconst_tail = mean_squared_error(Y_test[index_tail], Y_predict_unconst[index_tail]) 
    MSE_combine_tail = mean_squared_error(Y_test[index_tail], Y_predict_combine[index_tail]) 
    
    MSE_const_normal = mean_squared_error(Y_test[index_typical], Y_predict_const[index_typical]) 
    MSE_unconst_normal = mean_squared_error(Y_test[index_typical], Y_predict_unconst[index_typical]) 
    MSE_combine_normal = mean_squared_error(Y_test[index_typical], Y_predict_combine[index_typical]) 
     
    ratio_normal_const[iteration] = MSE_const_normal/MSE_unconst_normal - 1
    ratio_normal_combine[iteration] = MSE_combine_normal/MSE_unconst_normal - 1
    
    ratio_tail_const[iteration] = MSE_const_tail/MSE_unconst_tail - 1
    ratio_tail_combine[iteration] = MSE_combine_tail/MSE_unconst_tail - 1
    
    elapsed = time.time() - t
    print(elapsed)

ratio_normal_const_average = np.average(ratio_normal_const)    
ratio_normal_const_median = np.median(ratio_normal_const)    
ratio_normal_combine_average = np.average(ratio_normal_combine)    
ratio_normal_combine_median = np.median(ratio_normal_combine)    
        
ratio_tail_const_average = np.average(ratio_tail_const)    
ratio_tail_const_median = np.median(ratio_tail_const)    
ratio_tail_combine_average = np.average(ratio_tail_combine)    
ratio_tail_combine_median = np.median(ratio_tail_combine)    

print("const_typical_average = %.4f" %ratio_normal_const_average)
print("const_tail_average = %.4f" %ratio_tail_const_average)
print("combine_typical_average = %.4f" %ratio_normal_combine_average)
print("combine_tail_average = %.4f" %ratio_tail_combine_average)

print("const_typical_median = %.4f" %ratio_normal_const_median)
print("const_tail_median = %.4f" %ratio_tail_const_median)
print("combine_typical_median = %.4f" %ratio_normal_combine_median)
print("combine_tail_median = %.4f" %ratio_tail_combine_median)

rv = studentt(df=df, loc=0, scale=1)
z = np.linspace(rv.ppf(0.000001), rv.ppf(0.999999), 1000)
p_z = rv.pdf(z) 
log_p_z = np.log(p_z)

Z_tail_index = np.logical_or(Z.reshape(N) < -np.sqrt(1/alpha)*np.std(Z), Z.reshape(N) > np.sqrt(1/alpha)*np.std(Z))
Z_tail = np.zeros(N)
Z_tail[Z_tail_index] = 1

Z_reshaped = Z.reshape(N)

plt.figure()
plt.plot(z, p_z, label=r'$p(z)$', color = 'k')
plt.hist(Z_reshaped, color='gray', bins=10, density=True)
plt.yscale('log')
plt.xlabel('$z$')
plt.ylabel('Density')
plt.xticks([-10, -3*np.std(Z), 0, 3*np.std(Z), 10], ['-10', '-3$\sigma$', '0', '3$\sigma$', '10'])
plt.xlim([-20, 20])
plt.ylim([1e-5, 1])
plt.savefig("z_density.pdf", bbox_inches='tight')



#X_distance = np.zeros(N)
#for i in range(N):
##    X_distance[i] = np.sqrt(distance.mahalanobis(X[:, i], np.zeros(d), Sigma_x))
#    X_distance[i] = distance.mahalanobis(X[:, i], np.zeros(d), np.cov(X))

Z_distance = np.zeros(N)
for i in range(N):
#    X_distance[i] = np.sqrt(distance.mahalanobis(X[:, i], np.zeros(d), Sigma_x))
    Z_distance[i] = distance.mahalanobis(Z[:, i], np.zeros(q), np.cov(Z))
    
#plt.figure()
#plt.plot(X_distance, Z_tail, '.')

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(Z_distance.reshape(-1, 1), Z_tail)

prob_Y_X = clf.predict_proba(np.sort(Z_distance).reshape(-1, 1))

plt.figure()
plt.plot(Z_distance[Z_tail_index == 1], Z_tail[Z_tail_index == 1], 'r.', label = 'Tail', marker='o', markersize=8)
plt.plot(Z_distance[Z_tail_index == 0], Z_tail[Z_tail_index == 0], 'b.', label = 'Typical', marker='+', markersize=8)
plt.plot(np.sort(Z_distance), prob_Y_X[:, 1], 'k--', linewidth=2, label = 'Probability')
plt.yticks([0, 1], [0, 1])
plt.xlim([0, 30])
#plt.ylim([0, 1])
plt.xlabel('Mahalanobis distance between a realization of $z$ and $p(z)$')
plt.ylabel('Pr$(z \in Z_{alpha} | x)$')
plt.legend()
plt.savefig("logistic_regression.pdf", bbox_inches='tight')