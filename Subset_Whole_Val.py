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
from utils.custom_dataset import load_std_regress_data,CustomDataset  #load_dataset_custom
from utils.Create_Slices import get_slices
from model.LinearRegression import RegressionNet, LogisticNet
from model.Find_Fair_Subset import FindSubset

from torch.utils.data import DataLoader

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
delt = float(sys.argv[7])

sub_epoch = 5

batch_size = 12000

learning_rate = 0.001 #0.05 
#change = [250,650,1250,1950,4000]#,4200]

fullset, valset, testset = load_std_regress_data (datadir, data_name, True)

x_trn, y_trn = torch.from_numpy(fullset[0]).float().to(device),\
     torch.from_numpy(fullset[1]).float().to(device)
x_tst, y_tst = torch.from_numpy(testset[0]).float().to(device),\
     torch.from_numpy(testset[1]).float().to(device) 
x_val, y_val = torch.from_numpy(valset[0]).float().to(device),\
     torch.from_numpy(valset[1]).float().to(device) 

if data_name == "synthetic":
    all_logs_dir = './results/Whole/' + data_name +"_"+str(x_trn.shape[0])+'/' + str(fraction) +\
        '/' +str(delt) + '/' +str(select_every)
else:
    all_logs_dir = './results/Whole/' + data_name+'/' + str(fraction) +\
        '/' +str(delt) + '/' +str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_delta:' + str(delt) +\
    '_epochs:' + str(num_epochs) + '_selEvery:' + str(select_every)
print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)

#fullset, data_dims = load_dataset_custom(datadir, data_name, True) # valset, testset,

'''if data_name == 'Community_Crime':
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],device,3)

    change = [20,40,80,160]

elif data_name == 'census':
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],device)
    
    #x_val_list, y_val_list = x_val_list[:-1], y_val_list[:-1]
    #x_tst_list, y_tst_list = x_tst_list[:-1], y_tst_list[:-1]
    #val_classes, tst_classes = val_classes[:-1], tst_classes[:-1]

    rescale = np.linalg.norm(x_trn)
    x_trn = x_trn/rescale

    for j in range(len(x_val_list)):
        x_val_list[j] = x_val_list[j]/rescale
        x_tst_list[j] = x_tst_list[j]/rescale

    num_cls = 2

    change = [500]#[50,75,100,550] #[100,150,160,170,200]

elif data_name == 'OnlineNewsPopularity':
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],device)

    change = [100] #[100,150,160,170,200]

elif data_name == 'LawSchool':
    change = [100] #[100,150,160,170,200]

elif data_name == 'German_credit':
    change = [150,450,500] #[100,150,160,170,200]'''

#change = [50,100,200,550]

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)#,valset[0].shape)

train_batch_size = min(bud,1000)

print_every = 50

deltas = torch.tensor(delt) #torch.tensor([0.1 for _ in range(len(x_val_list))])

def train_model(func_name,start_rand_idxs=None, bud=None):

    idxs = start_rand_idxs

    criterion = nn.MSELoss()
    
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    main_model = RegressionNet(M)
   
    main_model = main_model.to(device)
    #criterion_sum = nn.MSELoss(reduction='sum')
    #main_optimizer = optim.SGD(main_model.parameters(), lr=learning_rate)
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change, gamma=0.5)

    if func_name == 'Random':
        print("Starting Random Run!")
    elif func_name == 'Random with Prior':
        print("Starting Random with Prior Run!")

    idxs.sort()
    np.random.seed(42)
    np_sub_idxs = np.array(idxs)
    np.random.shuffle(np_sub_idxs)
    loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
            transform=None),shuffle=False,batch_size=train_batch_size)

    for i in range(num_epochs):
        
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[idxs], y_trn[idxs]

        for batch_idx in list(loader_tr.batch_sampler):
            
            inputs, targets = loader_tr.dataset[batch_idx]
            main_optimizer.zero_grad()
            l2_reg = 0
            
            scores = main_model(inputs)
            for param in main_model.parameters():
                l2_reg += torch.norm(param)
            
            loss = criterion(scores, targets) +  reg_lambda*l2_reg *len(idxs)
            loss.backward()

            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            #print(main_optimizer.param_groups[0]['lr'])

    #tst_accuracy = torch.zeros(len(x_tst_list))
    #val_accuracy = torch.zeros(len(x_val_list))
    
    main_model.eval()
    with torch.no_grad():
        full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("Final SubsetTrn and FullTrn Loss:", sub_trn_loss.item(),full_trn_loss.item(),file=logfile)
        
        val_out = main_model(x_val)
        val_loss = criterion(val_out, y_val)

        outputs = main_model(x_tst)
        test_loss = criterion(outputs, y_tst)
        

    return [val_loss,test_loss] #[val_accuracy, val_classes, tst_accuracy, tst_classes]

def train_model_fair(func_name,start_rand_idxs=None, bud=None):

    sub_idxs = start_rand_idxs

    criterion = nn.MSELoss()
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    main_model = RegressionNet(M)

    #criterion_sum = nn.MSELoss(reduction='sum')
    
    alphas = torch.rand_like(deltas,requires_grad=True)
    #print(alphas)
    #alphas = torch.ones_like(deltas,requires_grad=True)
    '''dual_optimizer = optim.SGD([{'params': main_model.parameters()},
                {'params': alphas}], lr=learning_rate) #'''
    main_optimizer = torch.optim.Adam([
                {'params': main_model.parameters()},
                {'params': alphas,'lr':learning_rate}], lr=learning_rate) #{'params': alphas} #

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change,\
    #     gamma=0.5) #[e*2 for e in change]

    alphas.requires_grad = False

    #delta_extend = torch.repeat_interleave(deltas,val_size, dim=0)

    if func_name == 'Random':
        print("Starting Random with fairness Run!")
    elif func_name == 'Fair_subset':

        fsubset = FindSubset(x_trn, y_trn, x_val, y_val,main_model,main_optimizer,criterion,\
            device,deltas,learning_rate,reg_lambda)

        fsubset.precompute(int(num_epochs/4),sub_epoch)
        
        '''cached_state_dict = copy.deepcopy(main_model.state_dict())
        clone_dict = copy.deepcopy(main_model.state_dict())
        main_model.load_state_dict(cached_state_dict)'''
        
        print("Starting Subset of size ",fraction," with fairness Run!")

    sub_idxs.sort()
    np.random.seed(42)
    np_sub_idxs = np.array(sub_idxs)
    np.random.shuffle(np_sub_idxs)
    loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
            transform=None),shuffle=False,batch_size=train_batch_size)
    
    for i in range(num_epochs):

        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[sub_idxs], y_trn[sub_idxs]

        for batch_idx in list(loader_tr.batch_sampler):
            
            inputs, targets = loader_tr.dataset[batch_idx]

            main_optimizer.zero_grad()
            l2_reg = 0
            
            scores = main_model(inputs)
            for param in main_model.parameters():
                l2_reg += torch.norm(param)
            
            '''alpha_extend = torch.repeat_interleave(alphas,val_size, dim=0)
            val_scores = main_model(x_val_combined)
            constraint = criterion(val_scores, y_val_combined) - delta_extend
            multiplier = torch.dot(alpha_extend,constraint)'''

            '''constraint = torch.zeros(len(x_val_list))
            for j in range(len(x_val_list)):
                
                inputs_j, targets_j = x_val_list[j], y_val_list[j]
                scores_j = main_model(inputs_j)
                constraint[j] = criterion(scores_j, targets_j) - deltas[j]'''

            scores_val = main_model(x_val)
            constraint = criterion(scores_val, y_val) - deltas
            multiplier = alphas*constraint#torch.dot(alphas,constraint)

            loss = criterion(scores, targets) +  reg_lambda*l2_reg*len(sub_idxs) + multiplier
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()
            #main_optimizer.param_groups[1]['lr'] = learning_rate/2
            
            for param in main_model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True

            scores_val = main_model(x_val)
            constraint = criterion(scores_val, y_val)
            multiplier = -1.0*alphas*constraint#torch.dot(-1.0*alphas ,constraint)

            #print(alphas,constraint)
            
            #main_optimizer.zero_grad()
            multiplier.backward()
            
            main_optimizer.step()
            #scheduler.step()

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            #print(alphas)

            for param in main_model.parameters():
                param.requires_grad = True

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            #print(alphas,constraint)
            #print(criterion(scores, targets) , reg_lambda*l2_reg*len(idxs) ,multiplier)
            #print(main_optimizer.param_groups)#[0]['lr'])

        if ((i + 1) % select_every == 0) and func_name not in ['Full']:

            cached_state_dict = copy.deepcopy(main_model.state_dict())
            clone_dict = copy.deepcopy(main_model.state_dict())

            if func_name == 'Fair_subset':

                sub_idxs = fsubset.return_subset(clone_dict,sub_epoch,sub_idxs,alphas,bud)

            main_model.load_state_dict(cached_state_dict)
    
    #print(constraint)
    #print(alphas)
    main_model.eval()
    with torch.no_grad():
        full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[sub_idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[sub_idxs])
        print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item(),file=logfile)
        
        val_out = main_model(x_val)
        val_loss = criterion(val_out, y_val)

        outputs = main_model(x_tst)
        test_loss = criterion(outputs, y_tst)

    return [val_loss,test_loss]#[val_accuracy, val_classes, tst_accuracy, tst_classes]


starting = time.process_time() 
full_fair = train_model_fair('Random', np.random.choice(N, size=N, replace=False))
ending = time.process_time() 
print("Full with fairness training time ",ending-starting, file=logfile)

rand_idxs = list(np.random.choice(N, size=bud, replace=False))

starting = time.process_time() 
rand_fair = train_model_fair('Random',rand_idxs,bud)
ending = time.process_time() 
print("Random with fairness training time ",ending-starting, file=logfile)

starting = time.process_time() 
sub_fair = train_model_fair('Fair_subset', rand_idxs,bud)
ending = time.process_time() 
print("Subset of size ",fraction," with fairness training time ",ending-starting, file=logfile)

starting = time.process_time() 
full = train_model('Random', np.random.choice(N, size=N, replace=False))
ending = time.process_time() 
print("Full training time ",ending-starting, file=logfile)

starting = time.process_time() 
rand = train_model('Random',rand_idxs)
ending = time.process_time() 
print("Random training time ",ending-starting, file=logfile)

methods = [full_fair,rand_fair,sub_fair,full,rand]
methods_names=["Full with Constraints","Random with Constraints","Subset with Constraints","Full","Random"] 


for me in range(len(methods)):
    
    print("\n", file=logfile)
    print(methods_names[me],file=logfile)
    print("Validation Error","|",methods[me][0],"|",file=logfile)
    #print('---------------------------------------------------------------------',file=logfile)

    '''print('|Class | Error|',file=logfile)

    for a in range(len(x_val_list)):
        print("|",methods[me][1][a],"|",methods[me][0][a],"|",file=logfile)'''

    #print('---------------------------------------------------------------------',file=logfile)

    print("Test Error","|",methods[me][1],"|",file=logfile)
    print('---------------------------------------------------------------------',file=logfile)

    '''print('|Class | Error|',file=logfile)

    for a in range(len(x_tst_list)):
        print("|",methods[me][3][a],"|",methods[me][2][a],"|",file=logfile)'''

    #print('---------------------------------------------------------------------',file=logfile)
