import copy
import datetime
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from utils.custom_dataset import load_dataset_custom
from utils.Create_Slices import get_slices
from model.LinearRegression import RegressionNet, DualNet

import math
import random

torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Using Device:", device)

## Convert to this argparse
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])
reg_lambda = float(sys.argv[6])

batch_size = 12000

learning_rate = 0.05
all_logs_dir = './results/LR/' + data_name + '/' + str(fraction) + '/' + str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
           '_selEvery:' + str(select_every)
print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)

fullset, valset, testset, data_dims = load_dataset_custom(datadir, data_name, True)

x_val_list, y_val_list, val_classes = get_slices('Community_Crime',valset[0], valset[1],device)
x_tst_list, y_tst_list, tst_classes = get_slices('Community_Crime',testset[0], testset[1],device)

x_trn, y_trn = torch.from_numpy(fullset[0]).float().to(device),\
    torch.from_numpy(fullset[1]).float().to(device) #np.delete(fullset[0],protect_feature, axis=1)

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)

print_every = 50

deltas = torch.tensor([0.1 for _ in range(len(x_val_list))])

def train_model(func_name,start_rand_idxs=None, bud=None):

    main_model = RegressionNet(M)
    main_model = main_model.to(device)
    
    idxs = start_rand_idxs
    criterion = nn.MSELoss()
    main_optimizer = optim.SGD(main_model.parameters(), lr=learning_rate)

    if func_name == 'Random':
        print("Starting Random Run!")
    elif func_name == 'Random with Prior':
        print("Starting Random with Prior Run!")

    for i in range(num_epochs):
        
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        main_optimizer.zero_grad()
        l2_reg = 0
        
        scores = main_model(inputs)
        for param in main_model.parameters():
            l2_reg += torch.norm(param)
        
        loss = criterion(scores, targets) +  reg_lambda*l2_reg  #*len(idxs)
        loss.backward(retain_graph=True)
        main_optimizer.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())

    tst_accuracy = torch.zeros(len(x_tst_list))
    val_accuracy = torch.zeros(len(x_val_list))
    
    main_model.eval()
    with torch.no_grad():
        full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
        
        for j in range(len(x_val_list)):
            
            val_out = main_model(x_val_list[j])
            val_loss = criterion(val_out, y_val_list[j])
            val_accuracy[j] = val_loss            

        for j in range(len(x_tst_list)):
            
            outputs = main_model(x_tst_list[j])
            test_loss = criterion(outputs, y_tst_list[j])
            tst_accuracy[j] = test_loss

    return [val_accuracy, val_classes, tst_accuracy, tst_classes]

def train_model_fair(func_name,start_rand_idxs=None, bud=None):

    main_model = RegressionNet(M)
    main_model = main_model.to(device)

    dual_model = DualNet(len(x_val_list))
    dual_model = dual_model.to(device)
    
    idxs = start_rand_idxs
    criterion = nn.MSELoss()
    main_optimizer = optim.SGD(main_model.parameters(), lr=learning_rate)
    dual_optimizer = optim.SGD(dual_model.parameters(), lr=learning_rate)

    if func_name == 'Random':
        print("Starting Random Run!")
    elif func_name == 'Random with Prior':
        print("Starting Random with Prior Run!")

    for i in range(num_epochs):
        
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        main_optimizer.zero_grad()
        dual_optimizer.zero_grad()
        l2_reg = 0
        
        scores = main_model(inputs)
        for param in main_model.parameters():
            l2_reg += torch.norm(param)
        
        slice_losses = []
        for j in range(len(x_val_list)):
            
            inputs_j, targets_j = x_val_list[j], y_val_list[j]
            scores_j = main_model(inputs_j)
            slice_losses.append(criterion(scores_j, targets_j) - deltas[j])  

        constraint = torch.tensor(slice_losses).view(1,-1)

        loss = criterion(scores, targets) +  reg_lambda*l2_reg + dual_model(constraint) #*len(idxs)
        loss.backward(retain_graph=True)
        main_optimizer.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())

        
        #if i %2 == 0:
        loss_dual = -(criterion(scores, targets) +  reg_lambda*l2_reg + dual_model(constraint))
        loss_dual.backward()
        dual_optimizer.step()

    tst_accuracy = torch.zeros(len(x_tst_list))
    val_accuracy = torch.zeros(len(x_val_list))
    
    main_model.eval()
    with torch.no_grad():
        full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
        
        for j in range(len(x_val_list)):
            
            val_out = main_model(x_val_list[j])
            val_loss = criterion(val_out, y_val_list[j])
            val_accuracy[j] = val_loss            

        for j in range(len(x_tst_list)):
            
            outputs = main_model(x_tst_list[j])
            test_loss = criterion(outputs, y_tst_list[j])
            tst_accuracy[j] = test_loss

    return [val_accuracy, val_classes, tst_accuracy, tst_classes]


starting = time.process_time() 
full = train_model('Random', np.random.choice(N, size=N, replace=False))
ending = time.process_time() 
print("Full training time ",ending-starting, file=logfile)

starting = time.process_time() 
full_fair = train_model_fair('Random', np.random.choice(N, size=N, replace=False))
ending = time.process_time() 
print("Full with fairness training time ",ending-starting, file=logfile)

methods = [full,full_fair]
methods_names=["Full","Full with Fairness"] 

for me in range(len(methods)):

    print(methods_names[me],file=logfile)
    print("Validation Error",file=logfile)
    print('---------------------------------------------------------------------',file=logfile)

    print('|Class | Error|',file=logfile)

    for a in range(len(x_val_list)):
        print("|",methods[me][1][a],"|",methods[me][0][a],"|",file=logfile)

    print('---------------------------------------------------------------------',file=logfile)

    print("\n", file=logfile)
    print("Test Error",file=logfile)
    print('---------------------------------------------------------------------',file=logfile)

    print('|Class | Error|',file=logfile)

    for a in range(len(x_tst_list)):
        print("|",methods[me][3][a],"|",methods[me][2][a],"|",file=logfile)

    print('---------------------------------------------------------------------',file=logfile)

