import math
import numpy as np
import time
import torch
import torch.nn as nn
import copy

from utils.custom_dataset import CustomDataset_WithId, CustomDataset
from torch.utils.data import DataLoader

class FindSubset_Vect_Fair(object):
    def __init__(self, x_trn, y_trn, x_val, y_val,model,loss,device,delta,lr,lam,batch):
        
        self.x_trn = x_trn
        self.y_trn = y_trn
        #self.trn_batch = trn_batch

        self.x_val_list = x_val
        self.y_val_list = y_val

        self.model = model
        self.criterion = loss 
        self.device = device

        self.delta = delta
        self.lr = lr
        self.dual_lr = 0.1
        self.lam = lam
        #self.optimizer = optimizer
        self.batch_size = batch

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)


    def precompute(self,f_pi_epoch,p_epoch,alphas):#,budget):

        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

        print("starting Pre compute")
        
        #Compute F_phi
        for i in range(f_pi_epoch):
            
            main_optimizer.zero_grad()
            l2_reg = 0

            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            constraint = torch.zeros(len(self.x_val_list),device=self.device)
            for j in range(len(self.x_val_list)):
                
                inputs_j, targets_j = self.x_val_list[j], self.y_val_list[j]
                scores_j = self.model(inputs_j)
                constraint[j] = self.criterion(scores_j, targets_j) - self.delta[j]

            multiplier = torch.dot(alphas,torch.max(constraint,torch.zeros_like(constraint)))

            loss = multiplier
            #self.F_phi = loss.item()
            loss.backward()

            for p in filter(lambda p: p.grad is not None, self.model.parameters()):\
                 p.grad.data.clamp_(min=-.1, max=.1)
            
            main_optimizer.step()

            dual_optimizer.zero_grad()

            constraint = torch.zeros(len(self.x_val_list),device=self.device)
            for j in range(len(self.x_val_list)):
                
                inputs_j, targets_j = self.x_val_list[j], self.y_val_list[j]
                scores_j = self.model(inputs_j)
                constraint[j] = self.criterion(scores_j, targets_j) - self.delta[j]

            multiplier = torch.dot(-1.0*alphas,constraint)
            multiplier.backward()
            
            dual_optimizer.step()

            #print(i,alphas)

            for p in filter(lambda p: p.grad is not None, alphas):\
                 p.grad.data.clamp_(min=-.1, max=.1)

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            alphas.requires_grad = True
            
            if loss.item() <= 0.:
                break

        print("Finishing F phi")

        if loss.item() <= 0.:
            alphas = torch.zeros_like(alphas)

        device_new = self.device 

        self.F_values = torch.zeros(len(self.x_trn),device=self.device)

        l = [torch.flatten(p) for p in self.model.state_dict().values()]
        flat = torch.cat(l).detach().clone()
        
        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn, self.y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size)

        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        
        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, idxs = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ele_delta = self.delta.to(self.device)#.repeat(targets.shape[0]).to(self.device)
        
            weights = flat.repeat(targets.shape[0], 1)
            ele_alphas = alphas.detach().view(1,-1).repeat(targets.shape[0],1).to(self.device)
            
            #print(ele_alphas.shape)

            exp_avg_w = torch.zeros_like(weights)
            exp_avg_sq_w = torch.zeros_like(weights)

            exp_avg_a = torch.zeros_like(ele_alphas)
            exp_avg_sq_a = torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1))\
                ,dim=1)

            bias_correction1 = 1.0 
            bias_correction2 = 1.0 
            
            #print(p_epoch)

            for i in range(p_epoch):

                fin_val_loss_g = torch.zeros_like(weights).to(device_new)
                
                for j in range(len(self.x_val_list)):
                    
                    inputs_val, targets_val = self.x_val_list[j], self.y_val_list[j]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device)\
                        .view(-1,1)),dim=1).to(device_new)
                    #print(exten_val.shape)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                    #print(exten_val_y[0])
                
                    val_loss_p = 2*(torch.matmul(exten_val,torch.transpose(weights, 0, 1).to(device_new))\
                         - exten_val_y)

                    fin_val_loss_g += torch.mean(val_loss_p[:,:,None]*exten_val[:,None,:],dim=0)*(ele_alphas[:,j][:,None])

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val #mod_val,val_loss_g,
                    torch.cuda.empty_cache()

                trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                weight_grad = fin_trn_loss_g+ 2*self.lam*weights +fin_val_loss_g

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = (self.lr/10000) * math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                alpha_grad = torch.zeros_like(ele_alphas).to(device_new)
                
                for j in range(len(self.x_val_list)):
                    
                    inputs_val, targets_val = self.x_val_list[j], self.y_val_list[j]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device)\
                        .view(-1,1)),dim=1).to(device_new)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                
                    val_loss_p = torch.matmul(exten_val,torch.transpose(weights, 0, 1).to(device_new))\
                         - exten_val_y #
                    
                    alpha_grad[:,j] = torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta[j]

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val
                    torch.cuda.empty_cache()

                exp_avg_a.mul_(beta1).add_(1.0 - beta1, alpha_grad)
                exp_avg_sq_a.mul_(beta2).addcmul_(1.0 - beta2, alpha_grad, alpha_grad)
                denom = exp_avg_sq_a.sqrt().add_(dual_optimizer.param_groups[0]['eps'])
                
                ele_alphas.addcdiv_(step_size, exp_avg_a, denom)
                
                ele_alphas[ele_alphas < 0] = 0.
                
            val_losses = torch.zeros(targets.shape[0],device=self.device)
            for j in range(len(self.x_val_list)):
                    
                inputs_val, targets_val = self.x_val_list[j], self.y_val_list[j]
                inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                
                val_loss = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y

                val_losses += (torch.mean(val_loss*val_loss,dim=0)-ele_delta[j])*ele_alphas[:,j]
                
            reg = torch.sum(weights*weights,dim=1)
            trn_loss = torch.sum(exten_inp*weights,dim=1) - targets

            self.F_values[idxs] = trn_loss*trn_loss+ self.lam*reg + val_losses
                 

        idxs = torch.nonzero(self.F_values < 0)   
        print(idxs,len(idxs))
        
        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,alphas,budget,batch):

        torch.set_printoptions(precision=10)
        
        m_values = self.F_values.detach().clone() 
        
        self.model.load_state_dict(theta_init)

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=batch)    

        with torch.no_grad():

            l = [torch.flatten(p) for p in self.model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)

            for batch_idx in list(loader_tr.batch_sampler):
            
                inputs, targets, idxs = loader_tr.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                scores = torch.matmul(inputs,self.model.linear.weight.T)

                error = (scores.view(-1) + self.model.linear.bias) - targets

                m_values[np.array(curr_subset)[idxs]]= torch.sum(error*error) + self.lam*l2_reg

        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu().numpy())



