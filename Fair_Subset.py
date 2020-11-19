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
from model.Find_Fair_Subset import FindSubset

import math
import random

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

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
#change = [250,650,1250,1950,4000]#,4200]

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

fullset, data_dims = load_dataset_custom(datadir, data_name, True) # valset, testset,

if data_name == 'Community_Crime':
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],device,3)
else:
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],device)
    
    #x_val_list, y_val_list = x_val_list[:-1], y_val_list[:-1]
    #x_tst_list, y_tst_list = x_tst_list[:-1], y_tst_list[:-1]
    #val_classes, tst_classes = val_classes[:-1], tst_classes[:-1]

    change = [250,650,1250,1900,2150]
#
#x_tst_list, y_tst_list, tst_classes = get_slices('Community_Crime',testset[0], testset[1],4,device)

x_trn, y_trn = torch.from_numpy(x_trn).float().to(device),\
    torch.from_numpy(y_trn).float().to(device) #np.delete(fullset[0],protect_feature, axis=1)

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)#,valset[0].shape)

print_every = 50

deltas = torch.tensor([0.1 for _ in range(len(x_val_list))])

def train_model(func_name,start_rand_idxs=None, bud=None):

    main_model = RegressionNet(M)
    main_model = main_model.to(device)
    
    idxs = start_rand_idxs
    criterion = nn.MSELoss()
    #criterion_sum = nn.MSELoss(reduction='sum')
    #main_optimizer = optim.SGD(main_model.parameters(), lr=learning_rate)
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change, gamma=0.5)

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
        
        loss = criterion(scores, targets) +  reg_lambda*l2_reg *len(idxs)
        loss.backward()

        main_optimizer.step()
        scheduler.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            #print(main_optimizer.param_groups[0]['lr'])

    tst_accuracy = torch.zeros(len(x_tst_list))
    val_accuracy = torch.zeros(len(x_val_list))
    
    main_model.eval()
    with torch.no_grad():
        full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("Final SubsetTrn and FullTrn Loss:", sub_trn_loss.item(),full_trn_loss.item())
        
        for j in range(len(x_val_list)):

            print(val_classes[j],len(x_val_list[j]))
            
            val_out = main_model(x_val_list[j])
            val_loss = criterion(val_out, y_val_list[j])
            val_accuracy[j] = val_loss            
        print()
        for j in range(len(x_tst_list)):

            print(tst_classes[j],len(x_tst_list[j]))
            
            outputs = main_model(x_tst_list[j])
            test_loss = criterion(outputs, y_tst_list[j])
            tst_accuracy[j] = test_loss

    return [val_accuracy, val_classes, tst_accuracy, tst_classes]

def train_model_fair(func_name,start_rand_idxs=None, bud=None):

    main_model = RegressionNet(M)
    main_model = main_model.to(device)
    
    idxs = start_rand_idxs
    criterion = nn.MSELoss()
    #criterion_sum = nn.MSELoss(reduction='sum')
    
    #alphas = torch.rand_like(deltas,requires_grad=True) 
    alphas = torch.ones_like(deltas,requires_grad=True)
    '''dual_optimizer = optim.SGD([{'params': main_model.parameters()},
                {'params': alphas}], lr=learning_rate) #'''
    main_optimizer = torch.optim.Adam([
                {'params': main_model.parameters()},
                {'params': alphas,'lr':learning_rate}], lr=learning_rate) #{'params': alphas} #

    scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change,\
         gamma=0.5) #[e*2 for e in change]

    alphas.requires_grad = False

    val_size = torch.zeros(len(x_val_list),dtype=torch.long)
    for j in range(len(x_val_list)):
        val_size[j] = len(y_val_list[j])

    x_val_combined = torch.cat(x_val_list)
    y_val_combined = torch.cat(y_val_list)

    delta_extend = torch.repeat_interleave(deltas,val_size, dim=0)

    if func_name == 'Full':
        print("Starting Full with fairness Run!")
    elif func_name == 'Fair_subset':

        '''fsubset = FindSubset(x_trn, y_trn, x_val_combined, y_val_combined,main_model,criterion,\
            val_size,device,delta_extend)
        
        cached_state_dict = copy.deepcopy(main_model.state_dict())
        clone_dict = copy.deepcopy(main_model.state_dict())
        fsubset.precompute(clone_dict)
        model.load_state_dict(cached_state_dict)'''
        
        print("Starting Subset of size ",fraction," with fairness Run!")
    
    for i in range(num_epochs):

        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        main_optimizer.zero_grad()
        l2_reg = 0
        
        scores = main_model(inputs)
        for param in main_model.parameters():
            l2_reg += torch.norm(param)
        
        '''alpha_extend = torch.repeat_interleave(alphas,val_size, dim=0)
        val_scores = main_model(x_val_combined)
        constraint = criterion(val_scores, y_val_combined) - delta_extend
        multiplier = torch.dot(alpha_extend,constraint)'''

        constraint = torch.zeros(len(x_val_list))
        for j in range(len(x_val_list)):
            
            inputs_j, targets_j = x_val_list[j], y_val_list[j]
            scores_j = main_model(inputs_j)
            constraint[j] = criterion(scores_j, targets_j) - deltas[j]

        multiplier = torch.dot(alphas,constraint)

        loss = criterion(scores, targets) +  reg_lambda*l2_reg*len(idxs) + multiplier
        loss.backward()
        
        main_optimizer.step()
        scheduler.step()
        main_optimizer.param_groups[1]['lr'] = learning_rate

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            #print(main_optimizer.param_groups)#[0]['lr'])
        
        for param in main_model.parameters():
            param.requires_grad = False
        alphas.requires_grad = True

        constraint = torch.zeros(len(x_val_list))
        for j in range(len(x_val_list)):
            
            inputs_j, targets_j = x_val_list[j], y_val_list[j]
            scores_j = main_model(inputs_j)
            constraint[j] = criterion(scores_j, targets_j) - deltas[j]

        multiplier = torch.dot(alphas,-1.0*constraint)
        
        main_optimizer.zero_grad()
        multiplier.backward()
        
        main_optimizer.step()
        #scheduler.step()

        alphas.requires_grad = False
        alphas.clamp_(min=0.0)

        for param in main_model.parameters():
            param.requires_grad = True

        #if ((i + 1) % select_every == 0) and func_name not in ['Full']:

    tst_accuracy = torch.zeros(len(x_tst_list))
    val_accuracy = torch.zeros(len(x_val_list))
    
    print(constraint)
    print(alphas)
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
full_fair = train_model_fair('Random', np.random.choice(N, size=N, replace=False))
ending = time.process_time() 
print("Full with fairness training time ",ending-starting, file=logfile)

starting = time.process_time() 
full = train_model('Random', np.random.choice(N, size=N, replace=False))
ending = time.process_time() 
print("Full training time ",ending-starting, file=logfile)

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
