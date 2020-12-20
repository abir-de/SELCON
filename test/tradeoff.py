import numpy as np
import os
import subprocess
import sys
import time
import datetime

sys.path.append('../.') 
from utils.custom_dataset import load_dataset_custom, load_std_regress_data 
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
    #print(len(X))
    return (1.0 / X.shape[0]) *loss_fn(X, Y, beta) + lambd*regularizer(beta)#*len(X)

def mse(X, Y, beta):
    #print(X.shape)
    #print(loss_fn(X, Y, beta))
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta)#.value

def mse_value(X, Y, beta):
    #print(X.shape)
    #print(loss_fn(X, Y, beta))
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

np.random.seed(42)
device = "cpu"
print("Using Device:", device)

fraction = 1
#data_name = 'Community_Crime'
#data_name = 'OnlineNewsPopularity'
#data_name = 'census'
#data_name = 'LawSchool'
#data_name = "German_credit"
#data_name = 'cadata'
#data_name = 'abalone'
#data_name = 'cpusmall'
data_name = "synthetic"
#data_name = 'housing'
#data_name = "mg"
reg_lambda = 0.1

datadir = '../../Datasets/data/'+data_name+"/"

'''if data_name == 'Community_Crime':

    fullset, data_dims = load_dataset_custom(datadir, data_name, True)

    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],'cpu',3)

    x_val = np.concatenate(x_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    x_tst = np.concatenate(x_tst_list, axis=0)
    y_tst = np.concatenate(y_tst_list, axis=0)

elif data_name == 'census':

    fullset, data_dims = load_dataset_custom(datadir, data_name, True)

    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],'cpu')

    
    x_val = np.concatenate(x_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    x_tst = np.concatenate(x_tst_list, axis=0)
    y_tst = np.concatenate(y_tst_list, axis=0)

else:'''

fullset, valset, testset = load_std_regress_data (datadir, data_name, True)

#print(valset[0][0])

x_trn, y_trn = fullset[0].astype(np.float32), fullset[1].astype(np.float32)
x_tst, y_tst = testset[0].astype(np.float32), testset[1].astype(np.float32)
x_val, y_val = valset[0].astype(np.float32), valset[1].astype(np.float32)

x_trn, y_trn = x_trn.astype(np.float32), y_trn.astype(np.float32) #np.delete(fullset[0],protect_feature, axis=1)

#sc = StandardScaler()
#x_trn = sc.fit_transform(x_trn)

#rescale = np.linalg.norm(x_trn)
#x_trn = x_trn/rescale
#_val = x_val/rescale
#x_tst = x_tst/rescale


beta = cp.Variable(x_trn.shape[1])
lambd = cp.Parameter(nonneg=True)

np.random.seed(42)
#rand_idxs = list(np.random.choice(x_trn.shape[0], size=int(0.1*x_trn.shape[0]), replace=False))
rand_idxs = list(np.random.choice(x_trn.shape[0], size=int(x_trn.shape[0]), replace=False))
objective = cp.Minimize(objective_fn(x_trn[rand_idxs], y_trn[rand_idxs], beta, lambd))

#objective = cp.Minimize(cp.sum_squares(x_trn @ beta - y_trn) + lambd *cp.pnorm(beta, p=2)**2)

#deltas =[0.1]*len(x_val_list)
#deltas = [0.1026,0.1655,0.2442,0.1305,0.2651]

#lamda = np.concatenate((np.zeros(1),np.logspace(-8,-1,num=8,base=10),np.logspace(-8,-1,num=8,base=10)*5\
#    ,np.ones(1)))

#lamda = np.concatenate((np.zeros(1),np.logspace(-8,4,num=13,base=10), np.logspace(-8,4,num=13,base=10)*5))
#lamda.sort()
lamda = [0.01]
#print(lamda)

deltas = [i/100 for i in range(99,3,-3)]
deltas.insert(0,1)
#deltas = [i/100 for i in range(5,0,-1)] #deltas + [i/100 for i in range(5,0,-1)]
#deltas = [i/1000 for i in range(55,30,-5)]

#deltas = [i/100 for i in range(24,3,-3)]

sigma = 30
sigma_w = 10

'''for dl in deltas:

    if data_name == "synthetic":
        all_logs_dir = './results/CVX/'+data_name+"/w="+str(sigma_w)+'/delta_'+str(sigma)
    else:
        all_logs_dir = './results/CVX/'+data_name+'/delta'
    
    print(all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])
    path_logfile = os.path.join(all_logs_dir, data_name +'_delta='+str(dl)+'.txt')
    
    logfile = None
    print(dl)
    
    for reg_lambda in lamda:

        constraints = [mse(x_val, y_val, beta) <= dl]

        prob = cp.Problem(objective, constraints)

        lambd.value = reg_lambda
        # The optimal objective value is returned by `prob.solve()`.

        try:
            result = prob.solve()#verbose=True)
        except:
            print("An exception occurred")
            break
            #continue

        #print(beta.value)

        if beta.value is None:
            print(reg_lambda)
            break
            #continue
        elif reg_lambda == 0:
            logfile = open(path_logfile, 'w')
        val_error = mse_value(x_val, y_val, beta)
        tst_error = mse_value(x_tst, y_tst, beta)


        #print("Delta",end=" ")
        print(reg_lambda,end=",",file=logfile)

        #print("Test",end=" ")
        print("{0:.3f}".format(tst_error),end=",",file=logfile)

        #print("Validation",end=" ")
        print("{0:.3f}".format(val_error),end=",",file=logfile)


        print("{0:.3f}".format(objective.value),",","{0:.3f}".format(mse_value(x_trn, y_trn, beta)),\
            ",","{0:.3f}".format(reg_lambda*regularizer(beta).value),file=logfile) #*len(x_trn)*
        #for con in constraints:
        #   print(con.dual_value)

    if logfile is not None:
        logfile.close()
        logfile = None
    else:
        break'''

logfile = None
typeOf = "random_constrait"#"Full_no_constraint"

if data_name == "synthetic":
    all_logs_dir = './results/CVX/'+data_name+"/w="+str(sigma_w)+'/delta_'+str(sigma)
else:
    all_logs_dir = './results/CVX/'+data_name+'/delta'

starting = time.process_time() 

for dl in [1]:#deltas:
    
    print(all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])
    path_logfile = os.path.join(all_logs_dir, data_name +typeOf+'_delta='+str(dl)+'.txt')

    print(dl)
    
    constraints = [mse(x_val, y_val, beta) <= dl]

    prob = cp.Problem(objective)#, constraints)

    lambd.value = lamda[0]*len(x_trn[rand_idxs])
    # The optimal objective value is returned by `prob.solve()`.

    try:
        result = prob.solve()#verbose=True)
        if dl == 1:
            logfile = open(path_logfile, 'w')
    except:
        print("An exception occurred")

        if logfile is not None:
            logfile.close()
            logfile = None
        break

    if beta.value is None:
        break
    
    val_error = mse_value(x_val, y_val, beta)
    tst_error = mse_value(x_tst, y_tst, beta)

    #print(beta.value)

    #print("Delta",end=" ")
    print(dl,end=",",file=logfile)

    #print("Test",end=" ")
    print("{0:.3f}".format(tst_error),end=",",file=logfile)

    #print("Validation",end=" ")
    print("{0:.3f}".format(val_error),end=",",file=logfile)


    print("{0:.3f}".format(objective.value),",","{0:.3f}".format(mse_value(x_trn, y_trn, beta)),\
        ",","{0:.3f}".format(lambd.value*regularizer(beta).value),file=logfile) #*len(x_trn)*len(x_trn[rand_idxs])*
    #for con in constraints:
    #   print(con.dual_value)

ending = time.process_time() 
print("CVX time ",ending-starting)#, file=logfile)

'''all_logs_dir = './results/CVX/'+data_name
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
#path_logfile = os.path.join(all_logs_dir, data_name + '_lamda.txt')
path_logfile = os.path.join(all_logs_dir, data_name +'_delta='+dl+'.txt')
logfile = open(path_logfile, 'w')

for lam in [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]:

    prob = cp.Problem(objective)

    lambd.value = lam
    result = prob.solve()#verbose=True)

    val_error = mse_value(x_val, y_val, beta)
    tst_error = mse_value(x_tst, y_tst, beta)

    
    #print("Delta",end=" ")
    print("{0:.4f}".format(lam),end=",",file=logfile)
    
    #print("Test",end=" ")
    print("{0:.3f}".format(tst_error),end=",",file=logfile)

    #print("Validation",end=" ")
    print("{0:.3f}".format(val_error),end=",",file=logfile)


    print("{0:.3f}".format(objective.value),",","{0:.3f}".format(mse_value(x_trn, y_trn, beta)),\
        ",","{0:.3f}".format(lam*regularizer(beta).value),file=logfile) #*len(x_trn)*

logfile.close()'''


