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
from utils.custom_dataset import load_dataset_custom,CustomDataset
from utils.Create_Slices import get_slices
from model.LinearRegression import RegressionNet, LogisticNet
from model.Find_Fair_Subset import FindSubset_Vect_Fair,FindSubset_Fair
#from model.Fair_Subset2 import FindSubset_Vect_Fair

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

sub_epoch = 3 #5

batch_size = 4000#1000

learning_rate = 0.01 #0.05 

#change = [250,650,1250,1950,4000]#,4200]

#all_logs_dir = './results/Slice/' + data_name + '/' + str(fraction) + '/' + str(select_every)
all_logs_dir = './results/Slice/' + data_name+'/' + str(fraction) +\
        '/' +str(delt) + '/' +str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
           '_selEvery:' + str(select_every)

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

fullset, data_dims = load_dataset_custom(datadir, data_name, True) # valset, testset,

if data_name == 'Community_Crime':
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
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],device)

    change = [100] #[100,150,160,170,200]

elif data_name == 'German_credit':
    x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
        = get_slices(data_name,fullset[0], fullset[1],device)

    change = [150,450,500] #[100,150,160,170,200]
#
#x_tst_list, y_tst_list, tst_classes = get_slices('Community_Crime',testset[0], testset[1],4,device)

x_trn, y_trn = torch.from_numpy(x_trn).float().to(device),\
        torch.from_numpy(y_trn).float().to(device) #np.delete(fullset[0],protect_feature, axis=1)

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)#,valset[0].shape)

train_batch_size = min(bud,1000)

print_every = 50

deltas = torch.tensor([delt for _ in range(len(x_val_list))])

def weight_reset(m):
    torch.manual_seed(42)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_model(func_name,start_rand_idxs=None, bud=None):

    idxs = start_rand_idxs

    criterion = nn.MSELoss()
    
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    main_model = RegressionNet(M)
    main_model.apply(weight_reset)
   
    main_model = main_model.to(device)
    #criterion_sum = nn.MSELoss(reduction='sum')
    #main_optimizer = optim.SGD(main_model.parameters(), lr=learning_rate)
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change, gamma=0.5)
    scheduler = optim.lr_scheduler.StepLR(main_optimizer, step_size=1, gamma=0.1)

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

    stop_count = 0
    prev_loss = 1000
    prev_loss2 = 1000
    i =0
    mul=1
    lr_count = 0
    while(True):
        #for i in range(curr_epoch):#num_epochs):
        
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[idxs], y_trn[idxs]

        temp_loss = 0.

        for batch_idx in list(loader_tr.batch_sampler):
            
            inputs, targets = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(device), targets.to(device)
            main_optimizer.zero_grad()
            
            scores = main_model(inputs)

            '''l2_reg = 0
            for param in main_model.parameters():
                l2_reg += torch.norm(param)'''

            l = [torch.flatten(p) for p in main_model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)
            
            loss = criterion(scores, targets) +  reg_lambda*l2_reg*len(batch_idx)
            temp_loss += loss.item()
            loss.backward()

            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            print(prev_loss,temp_loss,mul)
            #print(main_optimizer.param_groups[0]['lr'])

        if abs(prev_loss - temp_loss) <= 1*mul or prev_loss2 == temp_loss:
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

        if abs(prev_loss - temp_loss) <= 1e-3 and stop_count >= 10:
            print(i)
            break 
        elif abs(prev_loss - temp_loss) <= 1e-3:
            stop_count += 1
        else:
            stop_count = 0

        if i>=2000:
            break

        #print(temp_loss,prev_loss)
        prev_loss2 = prev_loss
        prev_loss = temp_loss
        i+=1

    tst_accuracy = torch.zeros(len(x_tst_list))
    val_accuracy = torch.zeros(len(x_val_list))
    
    main_model.eval()
    l = [torch.flatten(p) for p in main_model.parameters()]
    flat = torch.cat(l)

    print(func_name,len(idxs),file=modelfile)
    print(flat,file=modelfile)

    no_red_error = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
        '''full_trn_out = main_model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = main_model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item(),file=logfile)'''
        
        
        for j in range(len(x_val_list)):
            
            val_out = main_model(x_val_list[j])
            val_loss = criterion(val_out, y_val_list[j])
            val_accuracy[j] = val_loss            

        tst_loss =[]
        for j in range(len(x_tst_list)):
            
            outputs = main_model(x_tst_list[j])
            temp = no_red_error(outputs, y_tst_list[j])

            tst_loss.append(temp)
            print(list(temp.cpu().numpy()),file=modelfile)
        
        mean_loss = []
        for j in range(len(tst_loss)):
            for k in range(j+1,len(tst_loss)): 

                for l in range(len(tst_loss[j])):
                    for m in range(len(tst_loss[k])):
                        mean_loss.append((tst_loss[j][l]-tst_loss[k][m])**2)

        tst_accuracy = torch.mean(torch.tensor(mean_loss))


    return [val_accuracy, val_classes, tst_accuracy]#, tst_classes]


def train_model_fair(func_name,start_rand_idxs=None, bud=None):

    sub_idxs = start_rand_idxs

    criterion = nn.MSELoss()
    '''if data_name == "census":
        main_model = LogisticNet(M)
    else:'''
    main_model = RegressionNet(M)
    main_model.apply(weight_reset)

    main_model = main_model.to(device)

    #criterion_sum = nn.MSELoss(reduction='sum')
    
    alphas = torch.rand_like(deltas,device=device,requires_grad=True)
    #print(alphas)
    #alphas = torch.ones_like(deltas,requires_grad=True)
    '''main_optimizer = optim.SGD([{'params': main_model.parameters()},
                {'params': alphas}], lr=learning_rate) #'''
    main_optimizer = torch.optim.Adam([
                {'params': main_model.parameters()}], lr=learning_rate)
                
    dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=learning_rate) #{'params': alphas} #'''

    #dual_optimizer = torch.optim.SGD([{'params': alphas}], lr=0.01)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change,\
    #     gamma=0.5) #[e*2 for e in change]
    scheduler = optim.lr_scheduler.StepLR(main_optimizer, step_size=1, gamma=0.1)

    #alphas.requires_grad = False

    #delta_extend = torch.repeat_interleave(deltas,val_size, dim=0)

    #for param in main_model.parameters():
    #    print(param)

    if func_name == 'Random':
        print("Starting Random with fairness Run!")
    elif func_name == 'Fair_subset':

        print("Starting Subset selection with fairness Run!")

        cached_state_dict = copy.deepcopy(main_model.state_dict())
        alpha_orig = copy.deepcopy(alphas)

        fsubset = FindSubset_Vect_Fair(x_trn, y_trn, x_val_list, y_val_list,main_model,criterion,\
            device,deltas,learning_rate,reg_lambda,batch_size) #*5

        #sub_idxs = fsubset.precompute(int(num_epochs/4),sub_epoch,alpha_orig,bud)
        fsubset.precompute(int(num_epochs/4),sub_epoch,alpha_orig)

        '''main_model.load_state_dict(cached_state_dict)
        alpha_orig = copy.deepcopy(alphas)

        fsubset_d = FindSubset_Fair(x_trn, y_trn, x_val_list, y_val_list,main_model,criterion,\
            device,deltas,learning_rate,reg_lambda)
        
        fsubset_d.F_values =  fsubset.F_values
        
        #fsubset_d.precompute(int(num_epochs/4),sub_epoch,alpha_orig,batch_size)

        main_model.load_state_dict(cached_state_dict)'''
        
        print("Starting Subset of size ",fraction," with fairness Run!")

    sub_idxs.sort()
    np.random.seed(42)
    np_sub_idxs = np.array(sub_idxs)
    np.random.shuffle(np_sub_idxs)
    loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
            transform=None),shuffle=False,batch_size=train_batch_size)

    stop_epoch = num_epochs
    
    #for i in range(num_epochs):
    stop_count = 0
    prev_loss = 1000
    prev_loss2 = 1000
    i =0
    mul = 1
    lr_count = 0
    while (True):

        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        #inputs, targets = x_trn[sub_idxs], y_trn[sub_idxs]

        temp_loss = 0.

        for batch_idx in list(loader_tr.batch_sampler):
            
            inputs_trn, targets_trn = loader_tr.dataset[batch_idx]
            inputs_trn, targets_trn = inputs_trn.to(device), targets_trn.to(device)

            main_optimizer.zero_grad()
            
            scores = main_model(inputs_trn)

            '''l2_reg = 0
            for param in main_model.parameters():
                l2_reg += torch.norm(param)'''

            l = [torch.flatten(p) for p in main_model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)

            '''alpha_extend = torch.repeat_interleave(alphas,val_size, dim=0)
            val_scores = main_model(x_val_combined)
            constraint = criterion(val_scores, y_val_combined) - delta_extend
            multiplier = torch.dot(alpha_extend,constraint)'''

            constraint = torch.zeros(len(x_val_list),device=device)
            for j in range(len(x_val_list)):
                
                inputs_j, targets_j = x_val_list[j], y_val_list[j]
                scores_j = main_model(inputs_j)
                constraint[j] = criterion(scores_j, targets_j) - deltas[j]

            multiplier = torch.dot(alphas,torch.max(constraint,torch.zeros_like(constraint)))

            loss = criterion(scores, targets_trn) + reg_lambda*l2_reg*len(batch_idx) \
                + torch.max(multiplier,torch.zeros_like(multiplier)) #
            temp_loss += loss.item()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            main_optimizer.step()
            #scheduler.step()
            #main_optimizer.param_groups[1]['lr'] = learning_rate/2
            
            '''for param in main_model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True'''

            constraint = torch.zeros(len(x_val_list),device=device)
            for j in range(len(x_val_list)):
                
                inputs_j, targets_j = x_val_list[j], y_val_list[j]
                scores_j = main_model(inputs_j)
                constraint[j] = criterion(scores_j, targets_j) - deltas[j]

            multiplier = torch.dot(-1.0*alphas,constraint)
            
            #print(alphas,constraint)
            dual_optimizer.zero_grad()

            #main_optimizer.state = state_orig
            multiplier.backward()
            dual_optimizer.step()
            #print(main_optimizer.param_groups)
            #scheduler.step()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, alphas):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            #alphas = alphas*((constraint >0).float())
            alphas.requires_grad = True
            #print(alphas)

            '''for param in main_model.parameters():
                param.requires_grad = True'''

        #print(alphas,constraint)

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            print(prev_loss,temp_loss)#,alphas,constraint)#,alphas)
            #print(main_optimizer.state)#.keys())
            #print(alphas,constraint)
            #print(criterion(scores, targets) , reg_lambda*l2_reg*len(idxs) ,multiplier)
            #print(main_optimizer.param_groups)#[0]['lr'])

        if ((i + 1) % select_every == 0) and func_name not in ['Full']:

            cached_state_dict = copy.deepcopy(main_model.state_dict())
            clone_dict = copy.deepcopy(cached_state_dict)
            
            

            alphas.requires_grad = False
            #alpha_orig = alpha_orig*((constraint >0).float())
            alphas = alphas*((constraint >0).float())
            alphas.requires_grad = True

            alpha_orig = copy.deepcopy(alphas)
            
            if func_name == 'Fair_subset':

                '''d_sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    batch_size)

                print(d_sub_idxs[:10])'''

                n_sub_idxs = fsubset.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    batch_size) #sub_epoch
                
                #print(sub_idxs[:10])
                #n_sub_idxs.sort()
                #print(n_sub_idxs[:10])

                '''clone_dict = copy.deepcopy(cached_state_dict)
                alpha_orig = copy.deepcopy(alphas)

                sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    train_batch_size)
                print(sub_idxs[:10])'''

                
                new_ele = set(n_sub_idxs).difference(set(sub_idxs))
                print(len(new_ele),0.1*bud)

                if len(new_ele) > 0.1*bud:
                    main_optimizer = torch.optim.Adam([
                    {'params': main_model.parameters()}], lr=max(main_optimizer.param_groups[0]['lr'],\
                        0.001))
                    
                    dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=learning_rate)

                    #mul=1
                    stop_count = 0
                    lr_count = 0

                #elif len(new_ele) == 0:
                #    select_every = select_every*2


                sub_idxs = n_sub_idxs

                sub_idxs.sort()
                np.random.seed(42)
                np_sub_idxs = np.array(sub_idxs)
                np.random.shuffle(np_sub_idxs)
                loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
                        transform=None),shuffle=False,batch_size=train_batch_size)

            main_model.load_state_dict(cached_state_dict)

        if abs(prev_loss - temp_loss) <= 1*mul or abs(temp_loss - prev_loss2) <= 1*mul:
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
        
        if (abs(prev_loss - temp_loss) <= 1e-3 or abs(temp_loss - prev_loss2) <= 1e-3) and\
             stop_count >= 10:
            print(i,prev_loss,temp_loss,constraint)
            break 
        elif abs(prev_loss - temp_loss) <= 1e-3 or abs(temp_loss - prev_loss2) <= 1e-3:
            #print(prev_loss,temp_loss)
            stop_count += 1
        else:
            stop_count = 0

        '''if torch.sum(((constraint >0).float())).item() <= 0 and stop_count >= 10: #10:
            print(i,constraint)
            break
        elif torch.sum(((constraint >0).float())).item() <= 0:
            #print(alphas,constraint,stop_count)
            stop_count += 1
        else:
            stop_count = 0'''

        if i>=2000:
            break

        #print(temp_loss,prev_loss)
        prev_loss2 = prev_loss
        prev_loss = temp_loss
        i+=1

    tst_accuracy = torch.zeros(len(x_tst_list))
    val_accuracy = torch.zeros(len(x_val_list))

    no_red_error = torch.nn.MSELoss(reduction='none')
    
    #print(constraint)
    #print(alphas)
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
        print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item(),file=logfile)'''
        
        
        for j in range(len(x_val_list)):
            
            val_out = main_model(x_val_list[j])
            val_loss = criterion(val_out, y_val_list[j])
            val_accuracy[j] = val_loss            

        tst_loss =[]
        for j in range(len(x_tst_list)):
            
            outputs = main_model(x_tst_list[j])
            temp = no_red_error(outputs, y_tst_list[j])

            tst_loss.append(temp)
            print(list(temp.cpu().numpy()),file=modelfile)
        
        mean_loss = []
        for j in range(len(tst_loss)):
            for k in range(j+1,len(tst_loss)): 

                for l in range(len(tst_loss[j])):
                    for m in range(len(tst_loss[k])):
                        mean_loss.append((tst_loss[j][l]-tst_loss[k][m])**2)

        tst_accuracy = torch.mean(torch.tensor(mean_loss))


    return [val_accuracy, val_classes, tst_accuracy]#, tst_classes]

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
full_fair = train_model_fair('Random', np.random.choice(N, size=N, replace=False))
ending = time.process_time() 
print("Full with Constraints training time ",ending-starting, file=logfile)

curr_epoch = 1000 #max(full_fair[2],rand_fair[2],sub_fair[2])

starting = time.process_time() 
full = train_model('Random', np.random.choice(N, size=N, replace=False),curr_epoch)
ending = time.process_time() 
print("Full training time ",ending-starting, file=logfile)

starting = time.process_time() 
rand = train_model('Random',rand_idxs,curr_epoch)
ending = time.process_time() 
print("Random training time ",ending-starting, file=logfile)

deltas = torch.ones_like(deltas)
starting = time.process_time() 
sub = train_model_fair('Fair_subset', rand_idxs,bud)
ending = time.process_time() 
print("Subset of size ",fraction," with fairness training time ",ending-starting, file=logfile)

methods = [rand_fair,sub_fair,full_fair,full,rand,sub] #
methods_names= ["Random with Constraints","Subset with Constraints","Full with Constraints","Full","Random","Subset"]


for me in range(len(methods)):

    print(methods_names[me],file=logfile)
    print("Validation Error",file=logfile)
    print('---------------------------------------------------------------------',file=logfile)

    print('|Class | Error|',file=logfile)

    for a in range(len(x_val_list)):
        print("|",methods[me][1][a],"|",methods[me][0][a].item(),"|",file=logfile)

    print('---------------------------------------------------------------------',file=logfile)

    print("\n", file=logfile)
    #print("Test Error",file=logfile)
    print('---------------------------------------------------------------------',file=logfile)

    #print('|Class | Error|',file=logfile)

    '''for a in range(len(x_tst_list)):
        print("|",methods[me][3][a],"|",methods[me][2][a].item(),"|",file=logfile)'''

    print("Test Error","|",methods[me][2].item(),"|",file=logfile)

    print('---------------------------------------------------------------------',file=logfile)
