import math
import numpy as np
import time
import torch
import torch.nn as nn
import copy

from utils.custom_dataset import CustomDataset_WithId, CustomDataset
from torch.utils.data import DataLoader


class FindSubset(object):
    def __init__(self, x_trn, y_trn, x_val, y_val,model,loss,device,delta,lr,lam):
        
        self.x_trn = x_trn
        self.y_trn = y_trn
        #self.trn_batch = trn_batch

        self.x_val = x_val
        self.y_val = y_val

        self.model = model
        self.criterion = loss 
        self.device = device

        self.delta = delta
        self.lr = lr
        self.lam = lam
        #self.optimizer = optimizer

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

    def precompute(self,f_pi_epoch,p_epoch,alphas,batch_size):

        #self.model.load_state_dict(theta_init)

        #for param in self.model.parameters():
        #    print(param)

        print("starting Pre compute")
        #alphas = torch.rand_like(self.delta,requires_grad=True) 

        #print(alphas)

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,40,100],\
        #    gamma=0.5) #[e*2 for e in change]

        '''optimizer = torch.optim.Adam([{'params': self.model.parameters()},\
            {'params': alphas,'lr':self.lr}], lr=self.lr)'''
        
        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr) 

        #alphas.requires_grad = False

        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,transform=None),\
            shuffle=False,batch_size=batch_size)

        #Compute F_phi
        for i in range(f_pi_epoch):
            
            main_optimizer.zero_grad()
            l2_reg = 0

            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            #scores_val = self.model(self.x_val)
            #constraint = self.criterion(scores_val, self.y_val) - self.delta
            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                val_out = self.model(inputs)
                constraint += self.criterion(val_out, targets)
            
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - self.delta
            multiplier = alphas*constraint #torch.dot(alphas,constraint)

            loss = multiplier
            self.F_phi = loss.item()
            loss.backward()
            
            main_optimizer.step()
            #scheduler.step()
            
            '''for param in self.model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True'''

            dual_optimizer.zero_grad()

            #scores_val = self.model(self.x_val)
            #constraint = self.criterion(scores_val, self.y_val) - self.delta
            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                val_out = self.model(inputs)
                constraint += self.criterion(val_out, targets)
            
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - self.delta
            multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
            
            multiplier.backward()
            
            dual_optimizer.step()

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            alphas.requires_grad = True
            #print(alphas)

            '''for param in self.model.parameters():
                param.requires_grad = True'''

        print("Finishing F phi")

        self.F_values = torch.zeros(len(self.x_trn))
        alphas_orig = copy.deepcopy(alphas)
        cached_state_dict = copy.deepcopy(self.model.state_dict())

        #print(cached_state_dict)
        #print(alphas_orig)

        for trn_id in range(len(self.x_trn)):
            alphas = copy.deepcopy(alphas_orig)
            self.model.load_state_dict(cached_state_dict)

            '''if trn_id in [2,1]:
                for param in self.model.parameters():
                    print(param)
                print(alphas)'''

            main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
            dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr) 

            for i in range(p_epoch):
                
                inputs_trn, targets_trn = self.x_trn[trn_id], self.y_trn[trn_id].view(-1)
                inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)
                main_optimizer.zero_grad()
                l2_reg = 0
                
                scores = self.model(inputs_trn)
                #for param in self.model.parameters():
                #    l2_reg += torch.norm(param)
                l = [torch.flatten(p) for p in self.model.parameters()]
                flat = torch.cat(l)
                l2_reg = torch.sum(flat*flat)
                
                constraint = 0.
                for batch_idx in list(loader_val.batch_sampler):
                        
                    inputs, targets = loader_val.dataset[batch_idx]
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                    val_out = self.model(inputs)
                    constraint += self.criterion(val_out, targets)
                
                constraint /= len(loader_val.batch_sampler)
                constraint = constraint - self.delta
                multiplier = alphas*constraint #torch.dot(alphas,constraint)

                #print(scores,targets)

                loss = self.criterion(scores, targets_trn) +  self.lam*l2_reg + multiplier
                '''if trn_id == 0:
                    print(scores)
                    print(self.criterion(scores, targets).item(),self.lam*l2_reg,multiplier.item())
                    print(inputs)'''
                #self.F_values[trn_id] = loss.item()
                loss.backward()

                '''if trn_id == 0:
                    for param in self.model.parameters():
                        print(param.grad)'''
                
                main_optimizer.step()
                #scheduler.step()

                '''if trn_id == 0:
                    for param in self.model.parameters():
                        print(param)'''

                '''for param in self.model.parameters():
                    param.requires_grad = False
                alphas.requires_grad = True'''

                dual_optimizer.zero_grad()

                constraint = 0.
                for batch_idx in list(loader_val.batch_sampler):
                        
                    inputs, targets = loader_val.dataset[batch_idx]
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                    val_out = self.model(inputs)
                    constraint += self.criterion(val_out, targets)
                
                constraint /= len(loader_val.batch_sampler)
                constraint = constraint - self.delta
                multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
                
                multiplier.backward()
                
                dual_optimizer.step()

                alphas.requires_grad = False
                alphas.clamp_(min=0.0)
                alphas.requires_grad = True
                '''if trn_id == 0:
                    print(alphas)'''

                '''for param in self.model.parameters():
                    param.requires_grad = True'''

            with torch.no_grad():

                scores = self.model(self.x_trn[trn_id])
                #for param in self.model.parameters():
                #    l2_reg += torch.norm(param)
                l = [torch.flatten(p) for p in self.model.parameters()]
                flat = torch.cat(l)
                l2_reg = torch.sum(flat*flat)

                constraint = 0.
                for batch_idx in list(loader_val.batch_sampler):
                        
                    inputs, targets = loader_val.dataset[batch_idx]
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                    val_out = self.model(inputs)
                    constraint += self.criterion(val_out, targets)
                
                constraint /= len(loader_val.batch_sampler)
                constraint = constraint - self.delta
                multiplier = alphas*constraint

                loss = self.criterion(scores, self.y_trn[trn_id].view(-1)) +\
                      self.lam*l2_reg + multiplier
            
                self.F_values[trn_id] = loss.item()

        #print(self.F_values[:10])

        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,alphas,budget,batch):

        m_values = copy.deepcopy(self.F_values) #torch.zeros(len(self.x_trn))
        
        self.model.load_state_dict(theta_init)

        #print(curr_subset)

        '''main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)'''

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=batch)

        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,transform=None),\
            shuffle=False,batch_size=batch)    

        with torch.no_grad():

            F_curr = 0.

            for batch_idx in list(loader_tr.batch_sampler):
            
                inputs, targets, _ = loader_tr.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                scores = self.model(inputs)
                print(self.criterion(scores, targets).item())

                F_curr += self.criterion(scores, targets).item() 

            F_curr /= len(loader_tr.batch_sampler)

            #print(F_curr)

            l2_reg = 0
            
            for param in self.model.parameters():
                l2_reg += torch.norm(param)

            valloss = 0.
            for batch_idx in list(loader_val.batch_sampler):
            
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                scores = self.model(inputs)
                valloss += self.criterion(scores, targets).item() 
            
            constraint = valloss/len(loader_val.batch_sampler) - self.delta
            multiplier = alphas*constraint #torch.dot(alphas,constraint)

            F_curr += (self.lam*l2_reg*len(curr_subset) + multiplier).item()

        #print(F_curr)

        alphas_orig = copy.deepcopy(alphas)
        cached_state_dict = copy.deepcopy(self.model.state_dict())

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,40,100],\
        #    gamma=0.5) #[e*2 for e in change]

        #alphas.requires_grad = False
        
        for sub_id in curr_subset:
            removed = copy.deepcopy(curr_subset)
            removed.remove(sub_id)

            alphas = copy.deepcopy(alphas_orig)
            self.model.load_state_dict(cached_state_dict)

            main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
            dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

            loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[removed], self.y_trn[removed],\
            transform=None),shuffle=False,batch_size=batch)

            for i in range(p_epoch):

                trn_loss = 0.
                for batch_idx in list(loader_tr.batch_sampler):
                
                    inputs_trn, targets_trn, _ = loader_tr.dataset[batch_idx]
                    inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)
                    main_optimizer.zero_grad()
                    
                    """l2_reg = 0
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)"""

                    l = [torch.flatten(p) for p in self.model.parameters()]
                    flat = torch.cat(l)
                    l2_reg = torch.sum(flat*flat)

                    scores = self.model(inputs_trn)

                    trn_loss += self.criterion(scores, targets_trn)
                    
                constraint = 0.
                for batch_idx in list(loader_val.batch_sampler):
                        
                    inputs, targets = loader_val.dataset[batch_idx]
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                    val_out = self.model(inputs)
                    constraint += self.criterion(val_out, targets)
                
                constraint /= len(loader_val.batch_sampler)
                constraint = constraint - self.delta
                multiplier = alphas*constraint #torch.dot(alphas,constraint)

                loss = trn_loss/len(loader_tr.batch_sampler) + self.lam*l2_reg*len(removed) + multiplier
                #m_values[sub_id] = F_curr - loss.item()
                loss.backward()

                '''if sub_id == curr_subset[0]:
                    for param in self.model.parameters():
                        print(param.grad)'''
                
                main_optimizer.step()
                #scheduler.step()

                '''for param in self.model.parameters():
                    param.requires_grad = False
                alphas.requires_grad = True'''

                dual_optimizer.zero_grad()

                constraint = 0.
                for batch_idx in list(loader_val.batch_sampler):
                        
                    inputs, targets = loader_val.dataset[batch_idx]
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                    val_out = self.model(inputs)
                    constraint += self.criterion(val_out, targets)
                
                constraint /= len(loader_val.batch_sampler)
                constraint = constraint - self.delta
                multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
                
                multiplier.backward()

                dual_optimizer.step()

                alphas.requires_grad = False
                alphas.clamp_(min=0.0)
                alphas.requires_grad = True
                #print(alphas)

                '''for param in self.model.parameters():
                    param.requires_grad = True'''

            with torch.no_grad():

                trn_loss = 0.
                for batch_idx in list(loader_tr.batch_sampler):
                
                    inputs_trn, targets_trn, idxs = loader_tr.dataset[batch_idx]
                    main_optimizer.zero_grad()

                    scores = self.model(inputs_trn)

                    trn_loss += self.criterion(scores, targets_trn).item()
                
                trn_loss/= len(loader_tr.batch_sampler)
                    
                constraint = 0.
                for batch_idx in list(loader_val.batch_sampler):
                        
                    inputs, targets = loader_val.dataset[batch_idx]
                
                    val_out = self.model(inputs)
                    constraint += self.criterion(val_out, targets).item()
                
                constraint /= len(loader_val.batch_sampler)
                constraint = constraint - self.delta
                multiplier = alphas*constraint #torch.dot(alphas,constraint)

                l = [torch.flatten(p) for p in self.model.parameters()]
                flat = torch.cat(l)
                l2_reg = torch.sum(flat*flat)

                m_values[sub_id] = F_curr - (trn_loss + self.lam*l2_reg*len(removed) + multiplier) #F_curr - 

        #print(curr_subset[:10])
        #print(F_curr)
        #print(m_values[curr_subset][:10])

        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu())

class FindSubset_Vect(object):
    def __init__(self, x_trn, y_trn, x_val, y_val,model,loss,device,delta,lr,lam,batch):
        
        self.x_trn = x_trn
        self.y_trn = y_trn
        #self.trn_batch = trn_batch

        self.x_val = x_val
        self.y_val = y_val

        self.model = model
        self.criterion = loss 
        self.device = device

        self.delta = delta
        self.lr = lr
        self.lam = lam
        #self.optimizer = optimizer
        self.batch_size = batch

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)


    def precompute(self,f_pi_epoch,p_epoch,alphas):

        #self.model.load_state_dict(theta_init)

        #for param in self.model.parameters():
        #    print(param)

        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

        print("starting Pre compute")
        #alphas = torch.rand_like(self.delta,requires_grad=True) 

        #print(alphas)

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,40,100],\
        #    gamma=0.5) #[e*2 for e in change]

        #alphas.requires_grad = False
        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,transform=None),\
            shuffle=False,batch_size=self.batch_size)
            

        #Compute F_phi
        for i in range(f_pi_epoch):
            
            main_optimizer.zero_grad()
            l2_reg = 0

            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            constraint = 0. 
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                val_out = self.model(inputs)
                constraint += self.criterion(val_out, targets)
                
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - self.delta
            multiplier = alphas*constraint #torch.dot(alphas,constraint)

            loss = multiplier
            self.F_phi = loss.item()
            loss.backward()
            
            main_optimizer.step()
            #scheduler.step()
            
            '''for param in self.model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True'''

            dual_optimizer.zero_grad()

            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                val_out = self.model(inputs)
                constraint += self.criterion(val_out, targets)
            
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - self.delta
            multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
            
            multiplier.backward()
            
            dual_optimizer.step()

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            alphas.requires_grad = True
            #print(alphas)

            '''for param in self.model.parameters():
                param.requires_grad = True'''

        print("Finishing F phi")

        self.F_values = torch.zeros(len(self.x_trn),device=self.device)
        #alphas_orig = copy.deepcopy(alphas)
        #cached_state_dict = copy.deepcopy(self.model.state_dict())

        l = [torch.flatten(p) for p in self.model.state_dict().values()]
        flat = torch.cat(l)
        #print(flat)
        
        #for param in self.model.parameters():
        #    print(param)
        #print(flat)
        #print(alphas)

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn, self.y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size)

        ele_delta = self.delta.repeat(min(self.batch_size,self.y_trn.shape[0])).to(self.device)

        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, idxs = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
        
            weights = flat.repeat(targets.shape[0], 1)
            ele_alphas = alphas.repeat(targets.shape[0]).to(self.device)
            #print(weights.shape)

            exp_avg_w = torch.zeros_like(weights)
            exp_avg_sq_w = torch.zeros_like(weights)

            exp_avg_a = torch.zeros_like(ele_alphas)
            exp_avg_sq_a = torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1))\
                ,dim=1)

            #print(exten_inp.shape)
            #print(inputs[0],idxs[0])#,exten_inp)
            #print(targets)

            #print(torch.sum(exten_inp*weights,dim=1)[0])

            '''trn_loss = torch.sum(exten_inp*weights,dim=1) - targets
            val_loss = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y
            reg = torch.sum(weights*weights,dim=1)'''

            #print(trn_loss.shape)

            #print((trn_loss*trn_loss)[0],self.lam*reg[0],\
            #    ((torch.mean(val_loss*val_loss,dim=0)-ele_delta)*ele_alphas)[0])

            bias_correction1 = 1.0 
            bias_correction2 = 1.0 

            for i in range(p_epoch):

                fin_val_loss_g = torch.zeros_like(weights)
                val_losses = torch.zeros_like(ele_delta)
                for batch_idx_val in list(loader_val.batch_sampler):
                    
                    inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                    #print(exten_val.shape)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                    #print(exten_val_y[0])
                
                    val_loss_p = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y
                    val_losses += torch.mean(val_loss_p*val_loss_p,dim=0)
                    val_loss_g = torch.unsqueeze(2*val_loss_p, dim=2).repeat(1,1,flat.shape[0])
                    #print(val_loss_g[0][0])

                    mod_val = torch.unsqueeze(exten_val, dim=1).repeat(1,targets.shape[0],1)
                    #print(mod_val[0])
                    fin_val_loss_g += torch.mean(val_loss_g*mod_val,dim=0)

                fin_val_loss_g /= len(loader_val.batch_sampler)

                trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                weight_grad = fin_trn_loss_g+ 2*self.lam*weights +fin_val_loss_g*ele_alphas[:,None]

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = self.lr * math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                #weights = weights - self.lr*(weight_grad)

                '''print(self.lr)
                print((fin_trn_loss_g+ 2*self.lam*weights +fin_val_loss_g*ele_alphas[:,None])[0])'''

                #print(weights[0])

                alpha_grad = val_losses/len(loader_val.batch_sampler)-ele_delta

                exp_avg_a.mul_(beta1).add_(1.0 - beta1, alpha_grad)
                exp_avg_sq_a.mul_(beta2).addcmul_(1.0 - beta2, alpha_grad, alpha_grad)
                denom = exp_avg_sq_a.sqrt().add_(dual_optimizer.param_groups[0]['eps'])
                ele_alphas.addcdiv_(step_size, exp_avg_a, denom)
                ele_alphas[ele_alphas < 0] = 0
                #print(ele_alphas[0])

                #ele_alphas = ele_alphas + self.lr*(torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta)

            val_losses = 0.
            for batch_idx_val in list(loader_val.batch_sampler):
                    
                inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                
                val_loss = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y

                val_losses+= torch.mean(val_loss*val_loss,dim=0)
            
            reg = torch.sum(weights*weights,dim=1)
            trn_loss = torch.sum(exten_inp*weights,dim=1) - targets

            #print(torch.sum(exten_inp*weights,dim=1)[0])
            #print((trn_loss*trn_loss)[0],self.lam*reg[0],\
            #    ((torch.mean(val_loss*val_loss,dim=0)-ele_delta)*ele_alphas)[0])

            self.F_values[idxs] = trn_loss*trn_loss+ self.lam*reg +\
                (val_losses/len(loader_val.batch_sampler)-ele_delta)*ele_alphas

        #print(self.F_values[:10])

        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,alphas,budget,batch):

        m_values = self.F_values.detach().clone() #torch.zeros(len(self.x_trn))
        
        self.model.load_state_dict(theta_init)

        #print(theta_init)
        #print(curr_subset)

        '''main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)'''

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=batch)

        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,transform=None),\
            shuffle=False,batch_size=batch)    

        with torch.no_grad():

            F_curr = 0.

            for batch_idx in list(loader_tr.batch_sampler):
            
                inputs, targets, _ = loader_tr.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                scores = self.model(inputs)
                #print(self.criterion(scores, targets).item())

                F_curr += self.criterion(scores, targets).item() 

            F_curr /= len(loader_tr.batch_sampler)
            #print(F_curr)

            l2_reg = 0
            
            for param in self.model.parameters():
                l2_reg += torch.norm(param)

            valloss = 0.
            for batch_idx in list(loader_val.batch_sampler):
            
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                scores = self.model(inputs)
                valloss += self.criterion(scores, targets).item() 
            
            constraint = valloss/len(loader_val.batch_sampler) - self.delta
            multiplier = alphas*constraint #torch.dot(alphas,constraint)

            F_curr += (self.lam*l2_reg*len(curr_subset) + multiplier).item()

        #print(F_curr)

        main_optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

        l = [torch.flatten(p) for p in self.model.state_dict().values()]
        flat = torch.cat(l)

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=self.batch_size)

        ele_delta = self.delta.repeat(min(self.batch_size,self.y_trn[curr_subset].shape[0])).to(self.device)

        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

        rem_len = (len(curr_subset)-1)

        b_idxs = 0

        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, _ = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
        
            weights = flat.repeat(targets.shape[0], 1)
            ele_alphas = alphas.repeat(targets.shape[0]).to(self.device)

            exp_avg_w = torch.zeros_like(weights)
            exp_avg_sq_w = torch.zeros_like(weights)

            exp_avg_a = torch.zeros_like(ele_alphas)
            exp_avg_sq_a = torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1)),dim=1)

            bias_correction1 = 1.0 
            bias_correction2 = 1.0 

            for i in range(p_epoch):

                fin_val_loss_g = torch.zeros_like(weights,device=self.device)
                val_losses = torch.zeros_like(ele_delta,device=self.device)
                for batch_idx_val in list(loader_val.batch_sampler):
                    
                    inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                    #print(exten_val.shape)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                    #print(exten_val_y[0])
                
                    val_loss_p = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y
                    val_losses += torch.mean(val_loss_p*val_loss_p,dim=0)
                    val_loss_g = torch.unsqueeze(2*val_loss_p, dim=2).repeat(1,1,flat.shape[0])
                    #print(val_loss_g[0][0])

                    mod_val = torch.unsqueeze(exten_val, dim=1).repeat(1,targets.shape[0],1)
                    #print(mod_val[0])
                    fin_val_loss_g += torch.mean(val_loss_g*mod_val,dim=0)

                fin_val_loss_g /= len(loader_val.batch_sampler)

                sum_fin_trn_loss_g = torch.zeros_like(weights)
                for batch_idx_trn in list(loader_tr.batch_sampler):
                    
                    inputs_trn, targets_trn,_ = loader_tr.dataset[batch_idx_trn]
                    inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)

                    exten_trn = torch.cat((inputs_trn,torch.ones(inputs_trn.shape[0],device=self.device).view(-1,1)),dim=1)
                    exten_trn_y = targets_trn.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                    #print(exten_val_y[0])
                
                    trn_loss_p = torch.matmul(exten_trn,torch.transpose(weights, 0, 1)) - exten_trn_y
                    sum_trn_loss_g = torch.unsqueeze(2*trn_loss_p, dim=2).repeat(1,1,flat.shape[0])
                    #print(val_loss_g[0][0])

                    mod_trn = torch.unsqueeze(exten_trn, dim=1).repeat(1,targets.shape[0],1)
                    sum_fin_trn_loss_g += torch.sum(sum_trn_loss_g*mod_trn,dim=0)

                #fin_trn_loss_g /= len(loader_tr.batch_sampler)

                trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                fin_trn_loss_g = (sum_fin_trn_loss_g - fin_trn_loss_g)/rem_len

                weight_grad = fin_trn_loss_g+ 2*rem_len*self.lam*weights +\
                    fin_val_loss_g*ele_alphas[:,None]

                #print(weight_grad[0])

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = self.lr * math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                #weights = weights - self.lr*(weight_grad)

                '''print(self.lr)
                print((fin_trn_loss_g+ 2*self.lam*weights +fin_val_loss_g*ele_alphas[:,None])[0])'''

                #print(weights[0])

                alpha_grad = val_losses/len(loader_val.batch_sampler)-ele_delta

                exp_avg_a.mul_(beta1).add_(1.0 - beta1, alpha_grad)
                exp_avg_sq_a.mul_(beta2).addcmul_(1.0 - beta2, alpha_grad, alpha_grad)
                denom = exp_avg_sq_a.sqrt().add_(dual_optimizer.param_groups[0]['eps'])
                ele_alphas.addcdiv_(step_size, exp_avg_a, denom)
                ele_alphas[ele_alphas < 0] = 0
                #print(ele_alphas[0])

                #ele_alphas = ele_alphas + self.lr*(torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta)

            val_losses = 0.
            for batch_idx_val in list(loader_val.batch_sampler):
                    
                inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                
                val_loss = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y

                val_losses+= torch.mean(val_loss*val_loss,dim=0)
            
            reg = torch.sum(weights*weights,dim=1)

            trn_losses = 0.
            for batch_idx_trn in list(loader_tr.batch_sampler):
                    
                inputs_trn, targets_trn,_ = loader_tr.dataset[batch_idx_trn]
                inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)

                exten_trn = torch.cat((inputs_trn,torch.ones(inputs_trn.shape[0],device=self.device).view(-1,1)),dim=1)
                exten_trn_y = targets_trn.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                #print(exten_val_y[0])
            
                trn_loss = torch.matmul(exten_trn,torch.transpose(weights, 0, 1)) - exten_trn_y
                
                trn_losses+= torch.sum(trn_loss*trn_loss,dim=0)

            trn_loss_ind = torch.sum(exten_inp*weights,dim=1) - targets

            trn_losses -= trn_loss_ind*trn_loss_ind
            #fin_trn_loss_g /= len(loader_tr.batch_sampler)
            #trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
            #fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]
            #trn_loss = torch.sum(exten_inp*weights,dim=1) - targets

            #print(torch.sum(exten_inp*weights,dim=1)[0])
            #print((trn_loss*trn_loss)[0],self.lam*reg[0],\
            #    ((torch.mean(val_loss*val_loss,dim=0)-ele_delta)*ele_alphas)[0])

            m_values[curr_subset[b_idxs*self.batch_size:(b_idxs+1)*self.batch_size]] =\
                F_curr -(trn_losses/rem_len+ self.lam*reg*rem_len\
                +(val_losses/len(loader_val.batch_sampler)-ele_delta)*ele_alphas)
            #F_curr -torch.tensor(F_curr).repeat(min(self.batch_size,self.y_trn[curr_subset].shape[0]))-
            b_idxs +=1

        #print(curr_subset[:10])
        #print(F_curr)
        #print(m_values[curr_subset][:10])
        
        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu())
        