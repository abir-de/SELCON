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

from sklearn.preprocessing import StandardScaler

import math
import random
import cvxpy as cp

def loss_fn(X, Y, beta):
    return cp.pnorm(X @ beta - Y, p=2)**2

def regularizer(beta):
    return cp.pnorm(beta, p=2)**2

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd*regularizer(beta) #*len(X)

def mse(X, Y, beta):
    #print(X.shape)
    #print(loss_fn(X, Y, beta))
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta)#.value

def mse_value(X, Y, beta):
    #print(X.shape)
    #print(loss_fn(X, Y, beta))
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

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
#data_name = 'Community_Crime'
#data_name = 'OnlineNewsPopularity'
#data_name = 'census'
#data_name = 'LawSchool'
data_name = "German_credit"
reg_lambda = 0.01

datadir = '../../Datasets/data/'+data_name+"/"

fullset, data_dims = load_dataset_custom(datadir, data_name, True) # valset, testset,

if data_name == 'Community_Crime':
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],'cpu',3)
else:
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],'cpu')

x_trn, y_trn = x_trn.astype(np.float32), y_trn.astype(np.float32) #np.delete(fullset[0],protect_feature, axis=1)

#sc = StandardScaler()
#x_trn = sc.fit_transform(x_trn)

rescale = np.linalg.norm(x_trn)
x_trn = x_trn/rescale

for i in range(len(x_val_list)):
    '''if data_name == 'OnlineNewsPopularity':
        x_val_list[i] = x_val_list[i].astype(np.float32)#sc.transform(x_val_list[i].numpy().astype(np.float32))
        x_tst_list[i] = x_tst_list[i].astype(np.float32)#sc.transform(x_tst_list[i].numpy().astype(np.float32))

    else:'''
    x_val_list[i] = x_val_list[i].numpy().astype(np.float32)/rescale
    #sc.transform(x_val_list[i].numpy().astype(np.float32))
    x_tst_list[i] = x_tst_list[i].numpy().astype(np.float32)/rescale
    #sc.transform(x_tst_list[i].numpy().astype(np.float32))
    
    y_tst_list[i] = y_tst_list[i].numpy().astype(np.float32)
    y_val_list[i] = y_val_list[i].numpy().astype(np.float32)

#print(x_trn[0])
#print(y_trn[0])

beta = cp.Variable(x_trn.shape[1])
lambd = cp.Parameter(nonneg=True)
objective = cp.Minimize(objective_fn(x_trn, y_trn, beta, lambd))

#objective = cp.Minimize(cp.sum_squares(x_trn @ beta - y_trn) + lambd *cp.pnorm(beta, p=2)**2)

#deltas =[0.1]*len(x_val_list)
#deltas = [0.1026,0.1655,0.2442,0.1305,0.2651]

all_logs_dir = './results/CVX' 
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')

for dl in [1,0.7,0.6,0.5,0.4,0.35,0.3,0.2,0.1,0.05,0.04,0.03]:
#[1,0.5,0.45,0.40,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.04,0.03,0.02,0.01]:
#[1,0.2651,0.25,0.22,0.21,0.2]:

    deltas =[dl]*len(x_val_list)
    constraints = []
    for i in range(len(x_val_list)):
        constraints = constraints + [mse(x_val_list[i], y_val_list[i], beta) <= deltas[i]]
        #constraints = constraints + [cp.sum_squares(x_val_list[i] @ beta - y_val_list[i]) <= 0.1]"""


    prob = cp.Problem(objective, constraints)

    lambd.value = reg_lambda
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()#verbose=True)

    #print(beta.value)
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    val_error =[]
    tst_error =[]
    for i in range(len(x_val_list)):
        val_error.append(mse_value(x_val_list[i], y_val_list[i], beta))
        tst_error.append(mse_value(x_tst_list[i], y_tst_list[i], beta))

    
    #print("Delta",end=" ")
    print("{0:.3f}".format(dl),end="\t")
    
    #print("Test",end=" ")
    print("{0:.3f}".format(sum(tst_error)),end="\t")
    #for i in range(len(x_val_list)):
    #    print(tst_error[i],end="\t")

    #print("Validation",end=" ")
    print("{0:.3f}".format(sum(val_error)),end="\t")
    #for i in range(len(x_val_list)):
    #    print(val_error[i],end="\t")


    print("{0:.3f}".format(objective.value),"\t","{0:.3f}".format(loss_fn(x_trn, y_trn, beta).value),\
        "\t","{0:.3f}".format(reg_lambda*regularizer(beta).value)) #*len(x_trn)*
    #for con in constraints:
    #   print(con.dual_value)

    print("Delta and training error:",dl,mse_value(x_trn, y_trn, beta),file=logfile)

    print("Validation and test error:",file=logfile)
    for i in range(len(x_val_list)):
        print(mse_value(x_val_list[i], y_val_list[i], beta), \
            mse_value(x_tst_list[i], y_tst_list[i], beta),file=logfile)
        


