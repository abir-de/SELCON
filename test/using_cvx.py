import numpy as np
import os
import subprocess
import sys
import time
import datetime

sys.path.append('../.') 
from utils.custom_dataset import load_dataset_custom
from utils.Create_Slices import get_slices
#from model.LinearRegression import RegressionNet, DualNet
#from model.Find_Fair_Subset import FindSubset

import math
import random
import cvxpy as cp

def loss_fn(X, Y, beta):
    return cp.pnorm(X @ beta - Y, p=2)**2

def regularizer(beta):
    return cp.pnorm(beta, p=2)**2

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd*len(X) * regularizer(beta)

def mse(X, Y, beta):
    #print(X.shape)
    #print(loss_fn(X, Y, beta))
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta)#.value

"""def generate_data(m=100, n=20, sigma=5):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    # Generate an ill-conditioned data matrix
    X = np.random.randn(m, n)
    # Corrupt the observations with additive Gaussian noise
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y

m = 100
n = 20
sigma = 5

X, Y = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

lambd.value = 0.01
problem.solve()
print(beta.value)"""

np.random.seed(42)
device = "cpu"
print("Using Device:", device)

## Convert to this argparse
'''datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])
reg_lambda = float(sys.argv[6])'''

fraction = 1
data_name = 'Community_Crime'
#data_name = 'census'
reg_lambda = 0.01

datadir = '../../Datasets/data/'+data_name+"/"

fullset, data_dims = load_dataset_custom(datadir, data_name, True) # valset, testset,

if data_name == 'Community_Crime':
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],'cpu',3)
else:
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],'cpu')

for i in range(len(x_val_list)):
    x_val_list[i] = x_val_list[i].numpy().astype(np.float32)
    y_val_list[i] = y_val_list[i].numpy().astype(np.float32)
    x_tst_list[i] = x_tst_list[i].numpy().astype(np.float32)
    y_tst_list[i] = y_tst_list[i].numpy().astype(np.float32)

x_trn, y_trn = x_trn.astype(np.float32), y_trn.astype(np.float32) #np.delete(fullset[0],protect_feature, axis=1)

#print(x_trn[0])
#print(y_trn[0])

beta = cp.Variable(x_trn.shape[1])
lambd = cp.Parameter(nonneg=True)
objective = cp.Minimize(objective_fn(x_trn, y_trn, beta, lambd))

#objective = cp.Minimize(cp.sum_squares(x_trn @ beta - y_trn) + lambd *cp.pnorm(beta, p=2)**2)

constraints = []
"""for i in range(len(x_val_list)):
    constraints = constraints + [mse(x_val_list[i], y_val_list[i], beta) <= 0.1]
    #constraints = constraints + [cp.sum_squares(x_val_list[i] @ beta - y_val_list[i]) <= 0.1]"""

prob = cp.Problem(objective, constraints)

lambd.value = reg_lambda
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()#verbose=True)

print(beta.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)

