import math
import numpy as np
import time
import torch
import torch.nn as nn
import copy

from utils.custom_dataset import CustomDataset_WithId, CustomDataset
from torch.utils.data import DataLoader

class FindSubset_Vect_Deep_rePre(object):
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

        #self.new_device = "cuda:1"

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)


    def precompute(self,f_pi_epoch,p_epoch,alphas):

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
        #for i in range(f_pi_epoch):

        prev_loss = 1000
        stop_count = 0
        i=0
        
        while(True):
            
            main_optimizer.zero_grad()
            
            '''l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)'''

            #l = [torch.flatten(p) for p in main_model.parameters()]
            #flat = torch.cat(l)
            #l2_reg = torch.sum(flat*flat)
            
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

            if loss.item() <= 0.:
                break

            #if i>= f_pi_epoch:
            #    break

            if abs(prev_loss - loss.item()) <= 1e-5 and stop_count >= 5:
                break 
            elif abs(prev_loss - loss.item()) <= 1e-5:
                stop_count += 1
            else:
                stop_count = 0

            prev_loss = loss.item()
            i+=1

            #if i % 50 == 0:
            #    print(loss.item(),alphas,constraint)

        print("Finishing F phi")

        if loss.item() <= 0.:
            alphas = torch.zeros_like(alphas)

        print(loss.item())

        l = [torch.flatten(p) for p in self.model.parameters()]
        #self.model.state_dict().values()]
        self.old_flat = torch.cat(l[2:]).detach().clone()

        self.old_alphas = alphas.detach().clone()
             

    def update_F_values(self,p_epoch,mean_val_x,mean_val_y): #loader_val,val_len):

        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)

        step_size = self.lr

        device_new = self.device #"cuda:2"#self.device #
        beta1,beta2 = main_optimizer.param_groups[0]['betas']

        self.F_values = torch.zeros(len(self.x_trn),device=self.device)

        loader_tr_ful = DataLoader(CustomDataset_WithId(self.x_trn, self.y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size*20)

        for batch_idx_trn in list(loader_tr_ful.batch_sampler):
                    
            inputs, targets,_ = loader_tr_ful.dataset[batch_idx_trn]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                out, l1 = self.model(inputs, last=True)
            
            if batch_idx_trn[0] == 0:

                mod_x_trn = l1
                mod_y_trn = targets
            else:

                batch_mod_x_trn = l1
                batch_mod_y_trn = targets

                mod_x_trn = torch.cat((mod_x_trn,batch_mod_x_trn), dim=0)
                mod_y_trn = torch.cat((mod_y_trn,batch_mod_y_trn), dim=0)

        loader_tr = DataLoader(CustomDataset_WithId(mod_x_trn , mod_y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size*50)
        
        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, idxs = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ele_delta = self.delta.repeat(targets.shape[0]).to(self.device)
        
            weights = self.old_flat.view(1,-1).repeat(targets.shape[0], 1)
            ele_alphas = self.old_alphas.detach().repeat(targets.shape[0]).to(self.device)
            #print(weights.shape)

            exp_avg_w = torch.zeros_like(weights)
            exp_avg_sq_w = torch.zeros_like(weights)

            #exp_avg_a = torch.zeros_like(ele_alphas)
            #exp_avg_sq_a = torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1))\
                ,dim=1)

            bias_correction1 = 1.0 
            bias_correction2 = 1.0 

            for i in range(p_epoch):

                trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                #no_bias = weights.clone()
                #no_bias[-1,:] = torch.zeros(weights.shape[0])
                
                weight_grad = fin_trn_loss_g+ 2*self.lam*\
                    torch.cat((weights[:,:-1], torch.zeros((weights.shape[0],1),device=self.device)),dim=1)
                #+fin_val_loss_g*ele_alphas[:,None]

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = step_size* math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                
            reg = torch.sum(weights[:,:-1]*weights[:,:-1],dim=1)
            trn_loss = torch.sum(exten_inp*weights,dim=1) - targets

            val_loss = torch.matmul(weights[:,:-1],mean_val_x) + weights[:,-1] - mean_val_y
                
            self.F_values[idxs] = trn_loss*trn_loss+ self.lam*reg \
                +torch.max(torch.zeros_like(ele_alphas),(val_loss*val_loss-ele_delta)*ele_alphas)

            #/val_len

        #print(self.F_values[:10])

        #self.F_values = self.F_values - max(loss.item(),0.) 

        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,alphas,budget,batch,\
        step,w_exp_avg,w_exp_avg_sq,a_exp_avg,a_exp_avg_sq):

        torch.set_printoptions(precision=10)
        
        self.model.load_state_dict(theta_init)

        loader_tr_ful = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=batch)

        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,transform=None),\
            shuffle=False,batch_size=batch)

        sum_error = torch.nn.MSELoss(reduction='sum')       

        for batch_idx_trn in list(loader_tr_ful.batch_sampler):
                    
            inputs, targets,_ = loader_tr_ful.dataset[batch_idx_trn]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                out, l1 = self.model(inputs, last=True)
            
            if batch_idx_trn[0] == 0:

                mod_x_trn = l1
                mod_y_trn = targets
            else:

                batch_mod_x_trn = l1
                batch_mod_y_trn = targets

                mod_x_trn = torch.cat((mod_x_trn,batch_mod_x_trn), dim=0)
                mod_y_trn = torch.cat((mod_y_trn,batch_mod_y_trn), dim=0)

        loader_tr = DataLoader(CustomDataset_WithId(mod_x_trn , mod_y_trn,\
            transform=None),shuffle=False,batch_size=batch)

        for batch_idx_val in list(loader_val.batch_sampler):
                    
            inputs_val, targets_val = loader_val.dataset[batch_idx_val]
            inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

            with torch.no_grad():
                out, l1 = self.model(inputs_val, last=True)
            
            if batch_idx_val[0] == 0:

                mod_x_val = l1
                #print(l1[0])
                mod_y_val = targets_val
            else:

                batch_mod_x_val = l1
                batch_mod_y_val = targets_val

                mod_x_val = torch.cat((mod_x_val,batch_mod_x_val), dim=0)
                mod_y_val = torch.cat((mod_y_val,batch_mod_y_val), dim=0)

        loader_val = DataLoader(CustomDataset(mod_x_val, mod_y_val,transform=None),\
            shuffle=False,batch_size=batch)

        mean_x_val = torch.mean(mod_x_val,dim=0)
        mean_y_val = torch.mean(mod_y_val)

        self.update_F_values(p_epoch,mean_x_val, mean_y_val) # loader_val)#
        #loader_val,len(mod_x_val))

        m_values = self.F_values.detach().clone()

        l = [torch.flatten(p) for p in self.model.parameters()]
        #self.model.state_dict().values()]
        flat = torch.cat(l[2:]).detach()

        with torch.no_grad():

            F_curr = 0.

            for batch_idx in list(loader_tr.batch_sampler):
            
                inputs, targets, _ = loader_tr.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                curr_error = torch.matmul(inputs,flat[:-1]) + flat[-1] - targets

                F_curr += torch.sum(curr_error*curr_error)

            #F_curr /= len(loader_tr.batch_sampler)
            #print(F_curr,end=",")

            '''l2_reg = 0
            
            for param in self.model.parameters():
                l2_reg += torch.norm(param)'''

            l2_reg = torch.sum(flat[:-1]*flat[:-1])

            #for p in self.model.parameters():
            #    print(p)

            valloss = torch.dot(flat[:-1],mean_x_val) + flat[-1] - mean_y_val
            
            for batch_idx in list(loader_val.batch_sampler):
            
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                curr_error = torch.matmul(inputs,flat[:-1]) + flat[-1] - targets
                valloss += torch.sum(curr_error*curr_error)
            
            constraint = valloss/len(mod_y_val) - self.delta #
            multiplier = alphas*constraint #torch.dot(alphas,constraint)
            
            F_curr += (self.lam*l2_reg*len(curr_subset) + multiplier).item()

        val_mul = multiplier.item()
        
        #print(self.lam*l2_reg*len(curr_subset), constraint, alphas)
        #print(F_curr)


        main_optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        
        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

        rem_len = (len(curr_subset)-1)

        b_idxs = 0

        device_new = self.device #"cuda:2" #self.device #

        step_size = self.lr

        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, _ = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ele_delta = self.delta.repeat(targets.shape[0]).to(self.device)
        
            weights = flat.repeat(targets.shape[0], 1)
            ele_alphas = alphas.detach().repeat(targets.shape[0]).to(self.device)

            exp_avg_w = w_exp_avg.repeat(targets.shape[0], 1)#torch.zeros_like(weights)
            exp_avg_sq_w = w_exp_avg_sq.repeat(targets.shape[0], 1) #torch.zeros_like(weights)

            exp_avg_a = a_exp_avg.repeat(targets.shape[0])#torch.zeros_like(ele_alphas)
            exp_avg_sq_a = a_exp_avg_sq.repeat(targets.shape[0]) #torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1)),dim=1)

            bias_correction1 = beta1**step#1.0 
            bias_correction2 = beta2**step#1.0 

            for i in range(p_epoch):

                fin_val_loss_g = torch.zeros_like(weights).to(device_new)
                #val_losses = torch.zeros_like(ele_delta).to(device_new)
                for batch_idx_val in list(loader_val.batch_sampler):
                    
                    inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device)\
                        .view(-1,1)),dim=1).to(device_new)
                    #print(exten_val.shape)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                    #print(exten_val_y[0])
                
                    val_loss_p = 2*(torch.matmul(exten_val,torch.transpose(weights, 0, 1).to(device_new))\
                         - exten_val_y) 

                    fin_val_loss_g += torch.sum(val_loss_p[:,:,None]*exten_val[:,None,:],dim=0)

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val #mod_val,val_loss_g,
                    torch.cuda.empty_cache()
                fin_val_loss_g = fin_val_loss_g/len(mod_y_val)

                #val_loss_g = 2*(torch.matmul(weights[:,:-1],mean_x_val) + weights[:,-1] - mean_y_val)

                #fin_val_loss_g = torch.cat((mean_x_val,torch.ones(1,device=self.device))).\
                #    repeat(targets.shape[0],1)*val_loss_g[:,None]
                #.to(self.device)

                sum_fin_trn_loss_g = torch.zeros_like(weights).to(device_new)
                for batch_idx_trn in list(loader_tr.batch_sampler):
                    
                    inputs_trn, targets_trn,_ = loader_tr.dataset[batch_idx_trn]
                    inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)

                    exten_trn = torch.cat((inputs_trn,torch.ones(inputs_trn.shape[0]\
                        ,device=self.device).view(-1,1)),dim=1).to(device_new)
                    exten_trn_y = targets_trn.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                    #print(exten_val_y[0])
                
                    sum_trn_loss_p = 2*(torch.matmul(exten_trn,torch.transpose(weights, 0, 1)\
                        .to(device_new)) - exten_trn_y)
                    #sum_trn_loss_g = torch.unsqueeze(trn_loss_p, dim=2).repeat(1,1,flat.shape[0])

                    #mod_trn = torch.unsqueeze(exten_trn, dim=1).repeat(1,targets.shape[0],1)
                    sum_fin_trn_loss_g += torch.sum(sum_trn_loss_p[:,:,None]*exten_trn[:,None,:],dim=0)

                    #print(sum_fin_trn_loss_g.shape)

                    del exten_trn,exten_trn_y,sum_trn_loss_p,inputs_trn, targets_trn #mod_trn,sum_trn_loss_g,
                    torch.cuda.empty_cache()

                #fin_trn_loss_g /= len(loader_tr.batch_sampler)
                sum_fin_trn_loss_g = sum_fin_trn_loss_g.to(self.device)

                trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                fin_trn_loss_g = (sum_fin_trn_loss_g - fin_trn_loss_g)/rem_len

                weight_grad = fin_trn_loss_g+ 2*rem_len*\
                    torch.cat((weights[:,:-1], torch.zeros((weights.shape[0],1),device=self.device)),dim=1)+\
                        fin_val_loss_g*ele_alphas[:,None]

                #print(weight_grad[0])

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = step_size* math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                #weights = weights - self.lr*(weight_grad)

                '''print(self.lr)
                print((fin_trn_loss_g+ 2*self.lam*weights +fin_val_loss_g*ele_alphas[:,None])[0])'''

                #print(weights[0])

                val_losses = torch.zeros_like(ele_delta).to(device_new)
                for batch_idx_val in list(loader_val.batch_sampler):
                    
                    inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device)\
                        .view(-1,1)),dim=1).to(device_new)
                    #print(exten_val.shape)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                    #print(exten_val_y[0])
                
                    val_loss_p = torch.matmul(exten_val,torch.transpose(weights, 0, 1).to(device_new))\
                         - exten_val_y #
                    val_losses += torch.sum(val_loss_p*val_loss_p,dim=0)

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val
                    torch.cuda.empty_cache()

                val_losses = val_losses.to(self.device)

                #val_loss = (torch.matmul(weights[:,:-1],mean_x_val) + weights[:,-1] - mean_y_val)
                #alpha_grad = val_loss*val_loss -ele_delta

                alpha_grad = val_losses/len(mod_y_val) -ele_delta
                
                exp_avg_a.mul_(beta1).add_(1.0 - beta1, alpha_grad)
                exp_avg_sq_a.mul_(beta2).addcmul_(1.0 - beta2, alpha_grad, alpha_grad)
                denom = exp_avg_sq_a.sqrt().add_(main_optimizer.param_groups[0]['eps'])
                ele_alphas.addcdiv_(step_size, exp_avg_a, denom)
                ele_alphas[ele_alphas < 0] = 0
                #print(ele_alphas[0])

                #ele_alphas = ele_alphas + self.lr*(torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta)

            val_losses = 0.
            for batch_idx_val in list(loader_val.batch_sampler):
                    
                inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                exten_val_y = targets_val.view(-1,1).repeat(1,targets.shape[0])
                
                #print(weights[:2])
                val_loss = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y

                val_losses+= torch.mean(val_loss*val_loss,dim=0)
            
            #val_loss = (torch.matmul(weights[:,:-1],mean_x_val) + weights[:,-1] - mean_y_val)
            #val_losses = val_loss*val_loss
            
            reg = torch.sum(weights[:,:-1]*weights[:,:-1],dim=1)

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
            
            '''m_values[curr_subset[b_idxs*self.batch_size:(b_idxs+1)*self.batch_size]] =\
                F_curr -(trn_losses + self.lam*reg*rem_len\
                +(val_losses/len(loader_val.batch_sampler)-ele_delta)*ele_alphas) #/rem_len'''

            abs_value = F_curr - (trn_losses + self.lam*reg*rem_len + \
                (val_losses/len(loader_val.batch_sampler) -ele_delta)*ele_alphas) #+ 1e-4 

            val_dec = (val_mul > (val_losses/len(loader_val.batch_sampler) -ele_delta)*ele_alphas).float()

            val_dec_ind = val_dec.nonzero().view(-1)

            neg_ind = ((abs_value ) < 0).nonzero().view(-1)

            abs_value [neg_ind] = torch.max(self.F_values)
            abs_value[val_dec_ind] = 0.0
            #self.F_values[torch.tensor(curr_subset)[b_idxs*self.batch_size:(b_idxs+1)*self.batch_size][val_dec_ind]]

            m_values[torch.tensor(curr_subset)[b_idxs*self.batch_size:(b_idxs+1)*self.batch_size]]\
                 = abs_value

            b_idxs +=1

        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu().numpy())


class FindSubset_Vect_TrnLoss_Deep(object):
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

        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

        print("starting Pre compute")
        
        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,transform=None),\
            shuffle=False,batch_size=self.batch_size)
     
        prev_loss = 1000
        stop_count = 0
        i=0
        
        while(True):
            
            main_optimizer.zero_grad()
            
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

            if loss.item() <= 0.:
                break

            #if i>= f_pi_epoch:
            #    break

            if abs(prev_loss - loss.item()) <= 1e-3 and stop_count >= 5:
                break 
            elif abs(prev_loss - loss.item()) <= 1e-3:
                stop_count += 1
            else:
                stop_count = 0

            prev_loss = loss.item()
            i+=1

            #if i % 50 == 0:
            #    print(loss.item(),alphas,constraint)

        print("Finishing F phi")

        if loss.item() <= 0.:
            alphas = torch.zeros_like(alphas)

        print(loss.item())
        
        l = [torch.flatten(p) for p in self.model.parameters()]
        #self.model.state_dict().values()]
        self.old_flat = torch.cat(l[2:]).detach().clone()

        self.old_alphas = alphas.detach().clone()
             

    def update_F_values(self,p_epoch,mean_val_x,mean_val_y): #loader_val,val_len):

        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)

        step_size = self.lr

        device_new = self.device #"cuda:2"#self.device #
        beta1,beta2 = main_optimizer.param_groups[0]['betas']

        self.F_values = torch.zeros(len(self.x_trn),device=self.device)

        loader_tr_ful = DataLoader(CustomDataset_WithId(self.x_trn, self.y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size*20)

        for batch_idx_trn in list(loader_tr_ful.batch_sampler):
                    
            inputs, targets,_ = loader_tr_ful.dataset[batch_idx_trn]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                out, l1 = self.model(inputs, last=True)
            
            if batch_idx_trn[0] == 0:

                mod_x_trn = l1
                mod_y_trn = targets
            else:

                batch_mod_x_trn = l1
                batch_mod_y_trn = targets

                mod_x_trn = torch.cat((mod_x_trn,batch_mod_x_trn), dim=0)
                mod_y_trn = torch.cat((mod_y_trn,batch_mod_y_trn), dim=0)

        loader_tr = DataLoader(CustomDataset_WithId(mod_x_trn , mod_y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size*50)
        
        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, idxs = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ele_delta = self.delta.repeat(targets.shape[0]).to(self.device)
        
            weights = self.old_flat.view(1,-1).repeat(targets.shape[0], 1)
            ele_alphas = self.old_alphas.detach().repeat(targets.shape[0]).to(self.device)
            #print(weights.shape)

            exp_avg_w = torch.zeros_like(weights)
            exp_avg_sq_w = torch.zeros_like(weights)

            #exp_avg_a = torch.zeros_like(ele_alphas)
            #exp_avg_sq_a = torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1))\
                ,dim=1)

            bias_correction1 = 1.0 
            bias_correction2 = 1.0 

            for i in range(p_epoch):

                trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                #no_bias = weights.clone()
                #no_bias[-1,:] = torch.zeros(weights.shape[0])
                
                weight_grad = fin_trn_loss_g+ 2*self.lam*\
                    torch.cat((weights[:,:-1], torch.zeros((weights.shape[0],1),device=self.device)),dim=1)
                #+fin_val_loss_g*ele_alphas[:,None]

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = step_size* math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                
            reg = torch.sum(weights[:,:-1]*weights[:,:-1],dim=1)
            trn_loss = torch.sum(exten_inp*weights,dim=1) - targets

            val_loss = torch.matmul(weights[:,:-1],mean_val_x) + weights[:,-1] - mean_val_y
                
            self.F_values[idxs] = trn_loss*trn_loss+ self.lam*reg \
                +torch.max(torch.zeros_like(ele_alphas),(val_loss*val_loss-ele_delta)*ele_alphas)

            #/val_len

        #print(self.F_values[:10])

        #self.F_values = self.F_values - max(loss.item(),0.) 

        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,budget,batch,\
        step,w_exp_avg,w_exp_avg_sq):#,alphas,a_exp_avg,a_exp_avg_sq):

        self.model.load_state_dict(theta_init)

        loader_tr_ful = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=batch)

        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,transform=None),\
            shuffle=False,batch_size=batch)

        sum_error = torch.nn.MSELoss(reduction='sum')       

        for batch_idx_trn in list(loader_tr_ful.batch_sampler):
                    
            inputs, targets,_ = loader_tr_ful.dataset[batch_idx_trn]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                out, l1 = self.model(inputs, last=True)
            
            if batch_idx_trn[0] == 0:

                mod_x_trn = l1
                mod_y_trn = targets
            else:

                batch_mod_x_trn = l1
                batch_mod_y_trn = targets

                mod_x_trn = torch.cat((mod_x_trn,batch_mod_x_trn), dim=0)
                mod_y_trn = torch.cat((mod_y_trn,batch_mod_y_trn), dim=0)

        loader_tr = DataLoader(CustomDataset_WithId(mod_x_trn , mod_y_trn,\
            transform=None),shuffle=False,batch_size=batch)

        for batch_idx_val in list(loader_val.batch_sampler):
                    
            inputs_val, targets_val = loader_val.dataset[batch_idx_val]
            inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

            with torch.no_grad():
                out, l1 = self.model(inputs_val, last=True)
            
            if batch_idx_val[0] == 0:

                mod_x_val = l1
                #print(l1[0])
                mod_y_val = targets_val
            else:

                batch_mod_x_val = l1
                batch_mod_y_val = targets_val

                mod_x_val = torch.cat((mod_x_val,batch_mod_x_val), dim=0)
                mod_y_val = torch.cat((mod_y_val,batch_mod_y_val), dim=0)

        loader_val = DataLoader(CustomDataset(mod_x_val, mod_y_val,transform=None),\
            shuffle=False,batch_size=batch)

        mean_x_val = torch.mean(mod_x_val,dim=0)
        mean_y_val = torch.mean(mod_y_val)

        self.update_F_values(p_epoch,mean_x_val, mean_y_val) # loader_val)#
        #loader_val,len(mod_x_val))

        m_values = self.F_values.detach().clone() #torch.zeros(len(self.x_trn))

        sum_error = torch.nn.MSELoss(reduction='sum')       

        with torch.no_grad():

            F_curr = 0.

            for batch_idx in list(loader_tr_ful.batch_sampler):
            
                inputs, targets, _ = loader_tr_ful.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                scores = self.model(inputs)
                #print(self.criterion(scores, targets).item())

                F_curr += sum_error(scores, targets).item() 

            #F_curr /= len(loader_tr.batch_sampler)
            #print(F_curr,end=",")

            l = [torch.flatten(p) for p in self.model.parameters()]
            flatt = torch.cat(l[2:])
            l2_reg = torch.sum(flatt[:-1]*flatt[:-1])

            F_curr += (self.lam*l2_reg*len(curr_subset)).item() #+ multiplier).item()

       
        #print(self.lam*l2_reg*len(curr_subset))

        main_optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        #dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

        l = [torch.flatten(p) for p in self.model.parameters()]
        #self.model.state_dict().values()]
        flat = torch.cat(l[2:]).detach()

        #ele_delta = self.delta.repeat(min(self.batch_size,self.y_trn[curr_subset].shape[0])).to(self.device)

        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

        rem_len = (len(curr_subset)-1)

        b_idxs = 0

        device_new = self.device #"cuda:2" #self.device #

        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, _ = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            weights = flat.repeat(targets.shape[0], 1)
            
            exp_avg_w = w_exp_avg.repeat(targets.shape[0], 1)
            exp_avg_sq_w = w_exp_avg_sq.repeat(targets.shape[0], 1) #torch.zeros_like(weights)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1)),dim=1)

            bias_correction1 = beta1**step#1.0 
            bias_correction2 = beta2**step#1.0 

            for i in range(p_epoch):

                sum_fin_trn_loss_g = torch.zeros_like(weights).to(device_new)
                for batch_idx_trn in list(loader_tr.batch_sampler):
                    
                    inputs_trn, targets_trn,_ = loader_tr.dataset[batch_idx_trn]
                    inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)

                    exten_trn = torch.cat((inputs_trn,torch.ones(inputs_trn.shape[0]\
                        ,device=self.device).view(-1,1)),dim=1).to(device_new)
                    exten_trn_y = targets_trn.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                    #print(exten_val_y[0])
                
                    sum_trn_loss_p = 2*(torch.matmul(exten_trn,torch.transpose(weights, 0, 1)\
                        .to(device_new)) - exten_trn_y)
                    #sum_trn_loss_g = torch.unsqueeze(trn_loss_p, dim=2).repeat(1,1,flat.shape[0])

                    #mod_trn = torch.unsqueeze(exten_trn, dim=1).repeat(1,targets.shape[0],1)
                    sum_fin_trn_loss_g += torch.sum(sum_trn_loss_p[:,:,None]*exten_trn[:,None,:],dim=0)

                    #print(sum_fin_trn_loss_g.shape)

                    del exten_trn,exten_trn_y,sum_trn_loss_p,inputs_trn, targets_trn #mod_trn,sum_trn_loss_g,
                    torch.cuda.empty_cache()

                #fin_trn_loss_g /= len(loader_tr.batch_sampler)
                sum_fin_trn_loss_g = sum_fin_trn_loss_g.to(self.device)

                trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                fin_trn_loss_g = (sum_fin_trn_loss_g - fin_trn_loss_g)/rem_len

                weight_grad = fin_trn_loss_g+ 2*rem_len*\
                    torch.cat((weights[:,:-1], torch.zeros((weights.shape[0],1),device=self.device)),dim=1)#+\
                #fin_val_loss_g*ele_alphas[:,None]

                #print(weight_grad[0])

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = (self.lr)* math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
            reg = torch.sum(weights[:,:-1]*weights[:,:-1],dim=1)

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
            
            abs_value = F_curr - (trn_losses + self.lam*reg*rem_len) 

            neg_ind = ((abs_value ) < 0).nonzero().view(-1)

            abs_value [neg_ind] = torch.max(self.F_values)

            m_values[torch.tensor(curr_subset)[b_idxs*self.batch_size:(b_idxs+1)*self.batch_size]]\
                 = abs_value

            b_idxs +=1

        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu().numpy())

