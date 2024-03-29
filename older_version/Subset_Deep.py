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
from utils.custom_dataset import load_std_regress_data,CustomDataset,load_dataset_custom
from utils.Create_Slices import get_slices

from model.TwoLayer import TwoLayerNet

from model.Deep_SELCON import FindSubset_Vect_Deep_rePre as FindSubset_Vect,\
    FindSubset_Vect_TrnLoss_Deep

from model.CRAIG import CRAIGStrategy 
from model.glister import Glister_Linear_SetFunction_RModular_Regression as GLISTER

from torch.utils.data import DataLoader

import math
import random

from utils.time_series import load_time_series_data

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
is_time = bool(int(sys.argv[8]))
psuedo_length = float(sys.argv[9])
if is_time:
    past_length = int(sys.argv[10])

sub_epoch = 3 #5

batch_size = 4000#1000

learning_rate = 0.01 #0.05 
hidden_units = 5

if is_time:
    fullset, valset, testset = load_time_series_data (datadir, data_name, past_length) #, sc_trans

    x_trn,y_trn =  torch.from_numpy(fullset[0]).float(),torch.from_numpy(fullset[1]).float()
    x_val,y_val =  torch.from_numpy(valset[0]).float(),torch.from_numpy(valset[1]).float()
    x_tst, y_tst = torch.from_numpy(testset[0]).float(),torch.from_numpy(testset[1]).float()

elif data_name in ['Community_Crime','census','LawSchool']:


    fullset, data_dims = load_dataset_custom(datadir, data_name, True)

    if data_name == 'Community_Crime':
        x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
            = get_slices(data_name,fullset[0], fullset[1],device,3)

        change = [20,40,80,160]

    elif data_name == 'census':
        x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
            = get_slices(data_name,fullset[0], fullset[1],device)
        
        rescale = np.linalg.norm(x_trn)
        x_trn = x_trn/rescale

        for j in range(len(x_val_list)):
            x_val_list[j] = x_val_list[j]/rescale
            x_tst_list[j] = x_tst_list[j]/rescale

        num_cls = 2

    elif data_name == 'LawSchool':
        x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
            = get_slices(data_name,fullset[0], fullset[1],device)

    x_trn, y_trn = torch.from_numpy(x_trn).float().to(device),torch.from_numpy(y_trn).float().to(device) 

    x_val,y_val = torch.cat(x_val_list,dim=0), torch.cat(y_val_list,dim=0)
    x_tst,y_tst = torch.cat(x_tst_list,dim=0), torch.cat(y_tst_list,dim=0)

else:
    fullset, valset, testset = load_std_regress_data (datadir, data_name, True)

    x_trn,y_trn =  torch.from_numpy(fullset[0]).float(),torch.from_numpy(fullset[1]).float()
    x_val,y_val =  torch.from_numpy(valset[0]).float(),torch.from_numpy(valset[1]).float()
    x_tst, y_tst = torch.from_numpy(testset[0]).float(),torch.from_numpy(testset[1]).float()


if data_name == "synthetic":
    all_logs_dir = './results/Deep/' + data_name +"_"+str(x_trn.shape[0])+'/' + str(fraction) +\
        '/' +str(delt) + '/' +str(select_every)
elif is_time:
    all_logs_dir = './results/Deep/' + data_name +"_"+str(past_length)+'/' + str(fraction) +\
        '/' +str(delt) + '/' +str(select_every)
else:
    all_logs_dir = './results/Deep/' + data_name+'/' + str(fraction) +\
        '/' +str(delt) + '/' +str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_delta:' + str(delt) +\
    '_epochs:' + str(num_epochs) + '_selEvery:' + str(select_every) + "_lambda" + str(reg_lambda)+\
        "_rand_sub_size"+str(psuedo_length)

path_logfile = os.path.join(all_logs_dir, data_name + '_model.txt')
modelfile = open(path_logfile, 'w')

print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)

print("=======================================", file=modelfile)
print(exp_name, str(exp_start_time), file=modelfile)

path_logfile = os.path.join(all_logs_dir, data_name + '_model.txt')
modelfile = open(path_logfile, 'w')


#N_val, M_val = x_val.shape
#print("Validation set Acccuracy",N_val,M_val)

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)#,valset[0].shape)

train_batch_size = min(bud,1000)

print_every = 50

deltas = torch.tensor(delt) #torch.tensor([0.1 for _ in range(len(x_val_list))])

def weight_reset(m):
    torch.manual_seed(42)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

def train_model(func_name,start_rand_idxs=None,curr_epoch=num_epochs, bud=None):

    idxs = start_rand_idxs

    criterion = nn.MSELoss()
    
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    #main_model = RegressionNet(M)

    main_model = TwoLayerNet(M,hidden_units)
    main_model.apply(weight_reset)

    #for p in main_model.parameters():
    #    print(p.data)
   
    main_model = main_model.to(device)
    #criterion_sum = nn.MSELoss(reduction='sum')
    #main_optimizer = optim.SGD(main_model.parameters(), lr=learning_rate)
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change, gamma=0.5)
    scheduler = optim.lr_scheduler.StepLR(main_optimizer, step_size=1, gamma=0.1)

    #print(idxs)

    idxs.sort()
    np.random.seed(42)
    np_sub_idxs = np.array(idxs)
    np.random.shuffle(np_sub_idxs)
    loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
            transform=None),shuffle=False,batch_size=train_batch_size)

    loader_full_tr = DataLoader(CustomDataset(x_trn, y_trn,transform=None),shuffle=False,\
        batch_size=train_batch_size)

    loader_val = DataLoader(CustomDataset(x_val, y_val,transform=None),shuffle=False,\
        batch_size=batch_size)

    if func_name == 'Random':
        print("Starting Random Run!")
    elif func_name == 'Random with Prior':
        print("Starting Random with Prior Run!")

    elif func_name == "Glister":
        glister = GLISTER(loader_full_tr, loader_val, main_model,learning_rate, device,'RGreedy',r=1)
        print("Starting glister of size ",fraction)

    elif func_name == "CRAIG":

        #bins = np.linspace(torch.min(y_trn).cpu().numpy(), torch.max(y_trn).cpu().numpy(), 16)
        #y_trn_bin = torch.tensor(np.digitize(y_trn.cpu().numpy(), bins)-1,device=device)

        y_trn_bin = torch.arange(20).repeat_interleave(int(y_trn.shape[0]/19))

        print(len(y_trn_bin))

        sorted_y, indices = torch.sort(y_trn)
                
        loader_full_tr_sort = DataLoader(CustomDataset(x_trn[indices], sorted_y,transform=None),\
            shuffle=False,batch_size=batch_size)

        craig = CRAIGStrategy(loader_full_tr_sort, loader_val,y_trn_bin[:y_trn.shape[0]], main_model,\
             device, 20, True,False, 'PerClass')
        
        print("Starting CRAIG of size ",fraction)

    elif func_name == 'Fair_subset':

        cached_state_dict = copy.deepcopy(main_model.state_dict())

        if psuedo_length == 1.0:
            sub_rand_idxs = [s for s in range(N)]
            current_idxs = idxs
        else:
            sub_rand_idxs = [s for s in range(N)]
            new_ele = set(sub_rand_idxs).difference(set(idxs))
            sub_rand_idxs = list(np.random.choice(list(new_ele), size=int(psuedo_length*N), replace=False))
            
            sub_rand_idxs = idxs + sub_rand_idxs

            current_idxs = [s for s in range(len(idxs))]

        fsubset_d = FindSubset_Vect_TrnLoss_Deep(x_trn[sub_rand_idxs], y_trn[sub_rand_idxs], x_val, y_val,main_model,\
            criterion,device,deltas,learning_rate,reg_lambda,batch_size)

        fsubset_d.precompute(int(num_epochs/4),sub_epoch,torch.randn_like(deltas,device=device))

        main_model.load_state_dict(cached_state_dict)
        
        print("Starting Subset of size ",fraction," with fairness Run!")

    stop_count = 0
    prev_loss = 1000
    prev_loss2 = 1000
    i =0
    mul=1
    lr_count = 0

    gamma = torch.ones(len(idxs),device=device)
    #while(True):
    for i in range(num_epochs):#curr_epoch):#num_epochs):
        
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[idxs], y_trn[idxs]

        temp_loss = 0.

        for batch_idx in list(loader_tr.batch_sampler):
            
            inputs, targets = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)
            main_optimizer.zero_grad()
                        
            scores = main_model(inputs)

            l2_reg = 0
            for param in main_model.parameters():
                l2_reg += torch.norm(param)

            '''l = [torch.flatten(p) for p in main_model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)'''
            
            if func_name == "CRAIG":
                loss = criterion(gamma[batch_idx]*scores, targets) +  reg_lambda*l2_reg*len(batch_idx)
            else:
                loss = criterion(scores, targets) +  reg_lambda*l2_reg*len(batch_idx)
            temp_loss += loss.item()
            loss.backward()

            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', temp_loss)
            print(prev_loss,temp_loss,mul)
            #print(main_optimizer.param_groups[0]['lr'])

        if ((i + 1) % select_every == 0) and func_name not in ['Random']:

            cached_state_dict = copy.deepcopy(main_model.state_dict())
            clone_dict = copy.deepcopy(cached_state_dict)

            if func_name == 'Fair_subset':

                fsubset_d.lr = min(main_optimizer.param_groups[0]['lr']*mul,1e-6)

                '''state_values = list(main_optimizer.state.values())
                step = state_values[0]['step']
                w_exp_avg = torch.cat((state_values[2]['exp_avg'].view(-1),state_values[3]['exp_avg']))
                w_exp_avg_sq = torch.cat((state_values[2]['exp_avg_sq'].view(-1),state_values[3]['exp_avg_sq']))'''

                step = 0
                w_exp_avg = torch.zeros(hidden_units+1,device=device)
                w_exp_avg_sq = torch.zeros(hidden_units+1,device=device)

                #print(exp_avg,exp_avg_sq)

                d_sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,current_idxs,\
                    bud,batch_size,step,w_exp_avg,w_exp_avg_sq)
                #torch.ones_like(deltas,device=device),a_exp_avg,a_exp_avg_sq)#,main_optimizer,dual_optimizer)

                '''clone_dict = copy.deepcopy(cached_state_dict)
                alpha_orig = copy.deepcopy(alphas)
                sub_idxs = fsubset.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    train_batch_size)
                print(sub_idxs[:10])'''

                current_idxs = d_sub_idxs
                #print(len(d_sub_idxs))

                d_sub_idxs = list(np.array(sub_rand_idxs)[d_sub_idxs])
                
            elif func_name == 'Glister':

                d_sub_idxs = glister.select(bud,clone_dict,N)

            elif func_name == "CRAIG":

                d_sub_idxs,gamma = craig.select(bud, clone_dict, 'stochastic')

                gamma = torch.tensor(gamma,device=device)

            new_ele = set(d_sub_idxs).difference(set(idxs))
            print(len(new_ele),0.1*bud)

            if len(new_ele) > 0.1*bud:
                main_optimizer = torch.optim.Adam([
                {'params': main_model.parameters()}], lr=main_optimizer.param_groups[0]['lr']*mul)
                #max(main_optimizer.param_groups[0]['lr'],0.001))

                #mul=1
                #stop_count = 0
                lr_count = 0
                
            idxs = d_sub_idxs

            idxs.sort()

            #print(idxs[:10])
            np.random.seed(42)
            np_sub_idxs = np.array(idxs)
            np.random.shuffle(np_sub_idxs)
            loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
                    transform=None),shuffle=False,batch_size=train_batch_size)

            main_model.load_state_dict(cached_state_dict)

        if abs(prev_loss - temp_loss) <= 1e-1*mul or prev_loss2 == temp_loss:
            #print(main_optimizer.param_groups[0]['lr'])
            lr_count += 1
            if lr_count == 10:
                #print(i,"Reduced")
                print(prev_loss,temp_loss,main_optimizer.param_groups[0]['lr'])
                scheduler.step()
                mul/=10
                lr_count = 0
        else:
            lr_count = 0

        '''if abs(prev_loss - temp_loss) <= 1e-3 and stop_count >= 5:
            print(i)
            break 
        elif abs(prev_loss - temp_loss) <= 1e-3:
            stop_count += 1
        else:
            stop_count = 0
       
        if i>=curr_epoch: #2000:
            break'''

        #print(temp_loss,prev_loss)
        prev_loss2 = prev_loss
        prev_loss = temp_loss
        #i+=1

    #tst_accuracy = torch.zeros(len(x_tst_list))
    #val_accuracy = torch.zeros(len(x_val_list))

    no_red_error = torch.nn.MSELoss(reduction='none')
    
    loader_tst = DataLoader(CustomDataset(x_tst, y_tst,transform=None),shuffle=False,\
        batch_size=batch_size)
    
    main_model.eval()

    l = [torch.flatten(p) for p in main_model.parameters()]
    flat = torch.cat(l)

    print(func_name,len(idxs),file=modelfile)
    print(flat,file=modelfile)
    
    with torch.no_grad():
        '''full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("\nFinal SubsetTrn and FullTrn Loss:", sub_trn_loss.item(),full_trn_loss.item(),file=logfile)'''
        
        #val_loss = 0.
        for batch_idx in list(loader_val.batch_sampler):
            
            inputs, targets = loader_val.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)
        
            val_out = main_model(inputs)
            '''if is_time:
                val_out = sc_trans.inverse_transform(val_out.cpu().numpy())
                val_out = torch.from_numpy(val_out).float()'''

            if batch_idx[0] == 0:
                e_val_loss = no_red_error(val_out, targets)

            else:
                batch_val_loss = no_red_error(val_out, targets)
                e_val_loss = torch.cat((e_val_loss, batch_val_loss),dim= 0)
        
        #val_loss /= len(loader_val.batch_sampler)
        val_loss = torch.mean(e_val_loss)
        print(list(e_val_loss.cpu().numpy()),file=modelfile)

        test_loss = 0.
        for batch_idx in list(loader_tst.batch_sampler):
            
            inputs, targets = loader_tst.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = main_model(inputs)
            '''if is_time:
                outputs = sc_trans.inverse_transform(outputs.cpu().numpy())
                outputs = torch.from_numpy(outputs).float()'''
            #test_loss += criterion(outputs, targets)

            if batch_idx[0] == 0:
                e_tst_loss = no_red_error(outputs, targets)

            else:
                batch_tst_loss = no_red_error(outputs, targets)
                e_tst_loss = torch.cat((e_tst_loss, batch_tst_loss),dim= 0)

        #test_loss /= len(loader_tst.batch_sampler)    
        test_loss = torch.mean(e_tst_loss)
        test_loss_std = torch.std(e_tst_loss)

        print(list(e_tst_loss.cpu().numpy()),file=modelfile)    

    return [val_loss,test_loss,test_loss_std] #[val_accuracy, val_classes, tst_accuracy, tst_classes]

def train_model_fair(func_name,start_rand_idxs=None, bud=None):

    sub_idxs = start_rand_idxs

    criterion = nn.MSELoss()
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    
    
    #main_model = RegressionNet(M)
    main_model = TwoLayerNet(M,hidden_units)
    main_model.apply(weight_reset)

    #for p in main_model.parameters():
    #    print(p.data)

    main_model = main_model.to(device)

    #criterion_sum = nn.MSELoss(reduction='sum')
    
    alphas = torch.randn_like(deltas,device=device) #+ 5. #,requires_grad=True)
    alphas.requires_grad = True
    #print(alphas)
    #alphas = torch.ones_like(deltas,requires_grad=True)
    '''main_optimizer = optim.SGD([{'params': main_model.parameters()},
                {'params': alphas}], lr=learning_rate) #'''
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)
    #[{'params': main_model.parameters()}], lr=learning_rate)
                
    dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=learning_rate) #{'params': alphas} #'''

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change,\
    #     gamma=0.5) #[e*2 for e in change]
    scheduler = optim.lr_scheduler.StepLR(main_optimizer, step_size=1, gamma=0.1)

    #alphas.requires_grad = False

    #delta_extend = torch.repeat_interleave(deltas,val_size, dim=0)    

    if func_name == 'Random':
        print("Starting Random with fairness Run!")
    elif func_name == 'Fair_subset':

        cached_state_dict = copy.deepcopy(main_model.state_dict())
        alpha_orig = copy.deepcopy(alphas)

        if psuedo_length == 1.0:
            sub_rand_idxs = [s for s in range(N)]
            current_idxs = sub_idxs
        else:
            sub_rand_idxs = [s for s in range(N)]
            new_ele = set(sub_rand_idxs).difference(set(sub_idxs))
            sub_rand_idxs = list(np.random.choice(list(new_ele), size=int(psuedo_length*N), replace=False))
            
            sub_rand_idxs = sub_idxs + sub_rand_idxs

            current_idxs = [s for s in range(len(sub_idxs))]

        fsubset_d = FindSubset_Vect(x_trn[sub_rand_idxs], y_trn[sub_rand_idxs], x_val, y_val,main_model,\
            criterion,device,deltas,learning_rate,reg_lambda,batch_size)

        fsubset_d.precompute(int(num_epochs/4),sub_epoch,alpha_orig)

        '''main_model.load_state_dict(cached_state_dict)
        alpha_orig = copy.deepcopy(alphas)
        fsubset = FindSubset(x_trn, y_trn, x_val, y_val,main_model,criterion,\
            device,deltas,learning_rate,reg_lambda)
        fsubset.precompute(int(num_epochs/4),sub_epoch,alpha_orig,batch_size)'''
        
        main_model.load_state_dict(cached_state_dict)
        
        print("Starting Subset of size ",fraction," with fairness Run!")
    
    sub_idxs.sort()
    np.random.seed(42)
    np_sub_idxs = np.array(sub_idxs)
    np.random.shuffle(np_sub_idxs)
    loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
            transform=None),shuffle=False,batch_size=train_batch_size)

    loader_val = DataLoader(CustomDataset(x_val, y_val,transform=None),shuffle=False,\
        batch_size=batch_size)

    loader_tst = DataLoader(CustomDataset(x_tst, y_tst,transform=None),shuffle=False,\
        batch_size=batch_size)

    stop_epoch = num_epochs
    
    #for i in range(num_epochs):
    stop_count = 0
    prev_loss = 1000
    prev_loss2 = 1000
    i =0
    mul = 1
    lr_count = 0
    #while (True):
    for i in range(num_epochs):

        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[sub_idxs], y_trn[sub_idxs]

        temp_loss = 0.

        starting = time.process_time() 

        for batch_idx_t in list(loader_tr.batch_sampler):
            
            inputs_trn, targets_trn = loader_tr.dataset[batch_idx_t]
            inputs_trn, targets_trn = inputs_trn.to(device), targets_trn.to(device)

            main_optimizer.zero_grad()
            
            scores_trn = main_model(inputs_trn)

            l2_reg = 0
            for param in main_model.parameters():
                l2_reg += torch.norm(param)

            '''l = [torch.flatten(p) for p in main_model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)'''

            #state_orig = copy.deepcopy(main_optimizer.state)
            
            '''alpha_extend = torch.repeat_interleave(alphas,val_size, dim=0)
            val_scores = main_model(x_val_combined)
            constraint = criterion(val_scores, y_val_combined) - delta_extend
            multiplier = torch.dot(alpha_extend,constraint)'''

            '''constraint = torch.zeros(len(x_val_list))
            for j in range(len(x_val_list)):
                
                inputs_j, targets_j = x_val_list[j], y_val_list[j]
                scores_j = main_model(inputs_j)
                constraint[j] = criterion(scores_j, targets_j) - deltas[j]'''

            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(device), targets.to(device)
            
                val_out = main_model(inputs)
                '''if is_time:
                    val_out = sc_trans.inverse_transform(val_out.cpu().numpy())
                    val_out = torch.from_numpy(val_out).float()'''
                constraint += criterion(val_out, targets)            
            
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - deltas
            multiplier = alphas*constraint*(float(constraint > 0)) #torch.dot(alphas,constraint)

            loss = criterion(scores_trn, targets_trn) + reg_lambda*l2_reg*len(batch_idx_t) + \
                multiplier #
            temp_loss += loss.item()
            loss.backward()

            #if i % print_every == 0:  
            #    print(criterion(scores_trn, targets_trn) , reg_lambda*l2_reg*len(batch_idx_t) ,multiplier)

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()
            #main_optimizer.param_groups[1]['lr'] = learning_rate/2
            
            '''for param in main_model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True'''

            dual_optimizer.zero_grad()

            #if constraint > 0:
            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(device), targets.to(device)
            
                val_out = main_model(inputs)
                '''if is_time:
                    val_out = sc_trans.inverse_transform(val_out.cpu().numpy())
                    val_out = torch.from_numpy(val_out).float()'''
                constraint += criterion(val_out, targets)
            
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - deltas
            multiplier = -1.0*alphas*constraint*(float(constraint > 0)) #torch.dot(-1.0*alphas ,constraint)

            #print(alphas,constraint)

            #main_optimizer.state = state_orig
            multiplier.backward()
            dual_optimizer.step()
            #print(main_optimizer.param_groups)
            #scheduler.step()`

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            alphas.requires_grad = True
            #print(alphas)

            '''for param in main_model.parameters():
                param.requires_grad = True'''

        #print(alphas,constraint)

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            print(prev_loss,temp_loss,constraint,alphas)
            #print(main_optimizer.state)#.keys())
            #print(alphas,constraint)
            #print(criterion(scores, targets) , reg_lambda*l2_reg*len(idxs) ,multiplier)
            #print(main_optimizer.param_groups)#[0]['lr'])


        if ((i + 1) % select_every == 0) and func_name not in ['Random']:

            cached_state_dict = copy.deepcopy(main_model.state_dict())
            clone_dict = copy.deepcopy(cached_state_dict)

            alpha_orig = alphas.detach().clone()#copy.deepcopy(alphas)

            '''alpha_orig.requires_grad = False
            alpha_orig = alpha_orig*((constraint >0).float())
            alpha_orig.requires_grad = True'''

            if func_name == 'Fair_subset':

                '''fsubset_d.lr = main_optimizer.param_groups[0]['lr']*mul
                state_values = list(main_optimizer.state.values())
                step = state_values[0]['step'] #0
                w_exp_avg = torch.cat((state_values[2]['exp_avg'].view(-1),state_values[3]['exp_avg']))
                w_exp_avg_sq = torch.cat((state_values[2]['exp_avg_sq'].view(-1),state_values[3]['exp_avg_sq']))
                
                state_values = list(dual_optimizer.state.values())
                
                a_exp_avg = state_values[0]['exp_avg'] 
                a_exp_avg_sq = state_values[0]['exp_avg_sq']'''
                
                if is_time:
                    fsubset_d.lr = min(main_optimizer.param_groups[0]['lr']*mul,1e-3)
                else:
                    fsubset_d.lr = min(main_optimizer.param_groups[0]['lr']*mul,1e-6)

                step = 0
                w_exp_avg = torch.zeros(hidden_units+1,device=device)
                w_exp_avg_sq = torch.zeros(hidden_units+1,device=device)
                a_exp_avg = torch.zeros(1,device=device)
                a_exp_avg_sq = torch.zeros(1,device=device)

                
                
                d_sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,current_idxs,alpha_orig,bud,\
                    batch_size,step,w_exp_avg,w_exp_avg_sq,a_exp_avg,a_exp_avg_sq)#,main_optimizer,dual_optimizer)

                '''clone_dict = copy.deepcopy(cached_state_dict)
                alpha_orig = copy.deepcopy(alphas)
                sub_idxs = fsubset.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    train_batch_size)
                print(sub_idxs[:10])'''

                current_idxs = d_sub_idxs

                d_sub_idxs = list(np.array(sub_rand_idxs)[d_sub_idxs])

                new_ele = set(d_sub_idxs).difference(set(sub_idxs))
                #print(len(new_ele),0.1*bud)

                if len(new_ele) > 0.1*bud:
                    main_optimizer = torch.optim.Adam([
                    {'params': main_model.parameters()}], lr=main_optimizer.param_groups[0]['lr']*mul)
                    #max(main_optimizer.param_groups[0]['lr'],0.001))
                    
                    dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=learning_rate)

                    #mul=1
                    stop_count = 0
                    lr_count = 0
                
                sub_idxs = d_sub_idxs

                sub_idxs.sort()

                print(sub_idxs[:10])
                np.random.seed(42)
                np_sub_idxs = np.array(sub_idxs)
                np.random.shuffle(np_sub_idxs)
                loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
                        transform=None),shuffle=False,batch_size=train_batch_size)

            main_model.load_state_dict(cached_state_dict)

        if abs(prev_loss - temp_loss) <= 1e-1*mul or abs(temp_loss - prev_loss2) <= 1e-1*mul:
            #print(main_optimizer.param_groups[0]['lr'])
            #print('lr',i)
            lr_count += 1
            if lr_count == 10:
                print(i,"Reduced",mul)
                #print(prev_loss,temp_loss,alphas)
                scheduler.step()
                mul/=10
                lr_count = 0
        else:
            lr_count = 0
        
        '''if (abs(prev_loss - temp_loss) <= 1e-3 or abs(temp_loss - prev_loss2) <= 1e-3) and\
             stop_count >= 5:
            print(i,prev_loss,temp_loss,constraint)
            break 
        elif abs(prev_loss - temp_loss) <= 1e-3 or abs(temp_loss - prev_loss2) <= 1e-3:
            #print(prev_loss,temp_loss)
            stop_count += 1
        else:
            stop_count = 0'''

        '''if constraint <= 0 and (stop_count >= 2 or (i + 1) % select_every == 0): #10:
            print(i,constraint)
            break
        elif constraint <= 0:
            #print(alphas,constraint,stop_count)
            stop_count += 1
        else:
            stop_count = 0'''


        '''if i>=2000:
            break'''
        
        prev_loss2 = prev_loss
        prev_loss = temp_loss
        #i +=1
        
    #print(constraint)
    #print(alphas)
    no_red_error = torch.nn.MSELoss(reduction='none')
    
    #loader_tst = DataLoader(CustomDataset(x_tst, y_tst,transform=None),shuffle=False,\
    #    batch_size=batch_size)
    
    main_model.eval()

    l = [torch.flatten(p) for p in main_model.parameters()]
    flat = torch.cat(l)

    print(func_name,len(sub_idxs),file=modelfile)
    print(flat,file=modelfile)
    
    with torch.no_grad():
        '''full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("\nFinal SubsetTrn and FullTrn Loss:", sub_trn_loss.item(),full_trn_loss.item(),file=logfile)'''
        
        #val_loss = 0.
        for batch_idx in list(loader_val.batch_sampler):
            

            inputs, targets = loader_val.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)
        
            val_out = main_model(inputs)
            '''if is_time:
                val_out = sc_trans.inverse_transform(val_out.cpu().numpy())
                val_out = torch.from_numpy(val_out).float()'''

            if batch_idx[0] == 0:
                e_val_loss = no_red_error(val_out, targets)

            else:
                batch_val_loss = no_red_error(val_out, targets)
                e_val_loss = torch.cat((e_val_loss, batch_val_loss),dim= 0)
        
        #val_loss /= len(loader_val.batch_sampler)
        val_loss = torch.mean(e_val_loss)
        print(list(e_val_loss.cpu().numpy()),file=modelfile)

        test_loss = 0.
        for batch_idx in list(loader_tst.batch_sampler):
            
            inputs, targets = loader_tst.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = main_model(inputs)
            '''if is_time:
                outputs = sc_trans.inverse_transform(outputs.cpu().numpy())
                outputs = torch.from_numpy(outputs).float()'''
            #test_loss += criterion(outputs, targets)

            if batch_idx[0] == 0:
                e_tst_loss = no_red_error(outputs, targets)

            else:
                batch_tst_loss = no_red_error(outputs, targets)
                e_tst_loss = torch.cat((e_tst_loss, batch_tst_loss),dim= 0)

        #test_loss /= len(loader_tst.batch_sampler)    
        test_loss = torch.mean(e_tst_loss)
        test_loss_std = torch.std(e_tst_loss)
        print(list(e_tst_loss.cpu().numpy()),file=modelfile)    

    return [val_loss,test_loss,test_loss_std,stop_epoch]#[val_accuracy, val_classes, tst_accuracy, tst_classes]

rand_idxs = list(np.random.choice(N, size=bud, replace=False))

starting = time.process_time() 
rand_fair = train_model_fair('Random',rand_idxs,bud)
ending = time.process_time() 
print("Random with Constraints training time ",ending-starting, file=logfile)

starting = time.process_time() 
sub_fair = train_model_fair('Fair_subset', rand_idxs,bud)
ending = time.process_time() 
print("Subset of size ",fraction," with fairness training time ",ending-starting, file=logfile)

starting = time.process_time() 
sub = train_model('Fair_subset', rand_idxs,bud=bud)
ending = time.process_time() 
print("Subset of size ",fraction,"training time ",ending-starting, file=logfile)

starting = time.process_time() 
full_fair = train_model_fair('Random', [i for i in range(N)])
ending = time.process_time() 
print("Full with Constraints training time ",ending-starting, file=logfile)

starting = time.process_time() 
full = train_model('Random', [i for i in range(N)],2000)
ending = time.process_time() 
print("Full training time ",ending-starting, file=logfile)

starting = time.process_time() 
rand = train_model('Random',rand_idxs,2000)
ending = time.process_time() 
print("Random training time ",ending-starting, file=logfile)

starting = time.process_time() 
facloc = train_model('CRAIG', rand_idxs,2000,bud=bud)
ending = time.process_time() 
print("CRAIG time ",ending-starting, file=logfile)

starting = time.process_time() 
glister = train_model('Glister', rand_idxs,2000,bud=bud)
ending = time.process_time() 
print("Glister time ",ending-starting, file=logfile)


methods = [rand_fair,sub_fair,sub,rand,facloc,glister,full,full_fair]
methods_names= ["Random with Constraints","Subset with Constraits","Subset","Random",\
    "CRAIG","GLISTER","Full","Full with Constraints"]

for me in range(len(methods)):
    
    print("\n", file=logfile)
    print(methods_names[me],file=logfile)
    print("Validation Error","|",methods[me][0].item(),"|",file=logfile)
   
    print("Test Error","|",methods[me][1].item(),"|",file=logfile)
    print("Test Error Std","|",methods[me][2].item(),"|",file=logfile)
    print('---------------------------------------------------------------------',file=logfile)

logfile.close()
modelfile.close()
