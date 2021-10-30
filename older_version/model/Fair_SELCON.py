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

        #self.model.load_state_dict(theta_init)

        '''for param in self.model.parameters():
            print(param)'''

        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)
        #dual_optimizer = torch.optim.SGD([{'params': alphas}], lr=self.dual_lr)

        print("starting Pre compute")
        #alphas = torch.rand_like(self.delta,requires_grad=True) 

        #print(alphas)

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,40,100],\
        #    gamma=0.5) #[e*2 for e in change]            

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
            #scheduler.step()
            
            #for param in self.model.parameters():
            #    param.requires_grad = False
            #alphas.requires_grad = True

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
            #print(alphas)

            #for param in self.model.parameters():
            #    param.requires_grad = True

            #if torch.sum(alphas).item() <= 0: 
            #    break
            if loss.item() <= 0.:
                break

        print("Finishing F phi")

        if loss.item() <= 0.:
            alphas = torch.zeros_like(alphas)

        #print(alphas)
        #print(loss.item())

        device_new = self.device #"cuda:2"#self.device #

        self.F_values = torch.zeros(len(self.x_trn),device=self.device)
        #alphas_orig = copy.deepcopy(alphas)
        #cached_state_dict = copy.deepcopy(self.model.state_dict())

        l = [torch.flatten(p) for p in self.model.state_dict().values()]
        flat = torch.cat(l).detach().clone()
        #print(flat)
        
        #for param in self.model.parameters():
        #    print(param)
        #print(flat)
        #print(alphas)

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn, self.y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size)

        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

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
                #val_losses = torch.zeros_like(ele_delta).to(device_new)
                #for batch_idx_val in list(loader_val.batch_sampler):
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

                    #print(torch.mean(val_loss_p[:,:,None]*exten_val[:,None,:],dim=0).shape)
                    #if i == 1:
                    #    print(ele_alphas[0],torch.mean(val_loss_p[:,:,None]*exten_val[:,None,:],dim=0)[0]\
                    #   ,(torch.mean(val_loss_p[:,:,None]*exten_val[:,None,:],dim=0)*(ele_alphas[:,j][:,None]))[0])
                    fin_val_loss_g += torch.mean(val_loss_p[:,:,None]*exten_val[:,None,:],dim=0)*(ele_alphas[:,j][:,None])

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val #mod_val,val_loss_g,
                    torch.cuda.empty_cache()

                #fin_val_loss_g /= len(self.x_val_list)
                #fin_val_loss_g = fin_val_loss_g.to(self.device)

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
                
                #weights = weights - self.lr*(weight_grad)

                '''print(self.lr)
                print((fin_trn_loss_g+ 2*self.lam*weights +fin_val_loss_g*ele_alphas[:,None])[0])'''

                #print(weights[129])

                #val_losses = torch.zeros_like(targets.shape[0]).to(device_new)
                alpha_grad = torch.zeros_like(ele_alphas).to(device_new)
                #print(alpha_grad.shape)
                for j in range(len(self.x_val_list)):
                    
                    inputs_val, targets_val = self.x_val_list[j], self.y_val_list[j]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device)\
                        .view(-1,1)),dim=1).to(device_new)
                    #print(exten_val.shape)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                    #print(exten_val_y[0])
                
                    val_loss_p = torch.matmul(exten_val,torch.transpose(weights, 0, 1).to(device_new))\
                         - exten_val_y #
                    
                    #print(torch.mean(val_loss_p*val_loss_p,dim=0)[129])#,torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta[j])
                    alpha_grad[:,j] = torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta[j]

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val
                    torch.cuda.empty_cache()

                    #val_losses = val_losses.to(self.device)
                    #alpha_grad = val_losses/len(loader_val.batch_sampler)-ele_delta
                #print(alpha_grad[129])
                exp_avg_a.mul_(beta1).add_(1.0 - beta1, alpha_grad)
                exp_avg_sq_a.mul_(beta2).addcmul_(1.0 - beta2, alpha_grad, alpha_grad)
                denom = exp_avg_sq_a.sqrt().add_(dual_optimizer.param_groups[0]['eps'])
                #print(ele_alphas[0])
                ele_alphas.addcdiv_(step_size, exp_avg_a, denom)
                
                #ele_alphas = ele_alphas + self.dual_lr*(alpha_grad)
                ele_alphas[ele_alphas < 0] = 0.
                #print(ele_alphas[0])

                #ele_alphas = ele_alphas + self.lr*(torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta)

            #print(ele_alphas[ele_alphas < 0])
            val_losses = torch.zeros(targets.shape[0],device=self.device)
            for j in range(len(self.x_val_list)):
                    
                inputs_val, targets_val = self.x_val_list[j], self.y_val_list[j]
                inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                
                val_loss = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y

                val_losses += (torch.mean(val_loss*val_loss,dim=0)-ele_delta[j])*ele_alphas[:,j]
                #print(val_losses[129])
            
            #print(ele_alphas[ele_alphas>0])
            reg = torch.sum(weights*weights,dim=1)
            trn_loss = torch.sum(exten_inp*weights,dim=1) - targets

            self.F_values[idxs] = trn_loss*trn_loss+ self.lam*reg + val_losses

        idxs = torch.nonzero(self.F_values < 0)   
        
        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,alphas,budget,batch):

        torch.set_printoptions(precision=10)
        
        m_values = self.F_values.detach().clone() 
        
        self.model.load_state_dict(theta_init)

        l = [torch.flatten(p) for p in self.model.state_dict().values()]
        flat_orig = torch.cat(l).detach()

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=batch)

        sum_error = torch.nn.MSELoss(reduction='sum')      


        with torch.no_grad():

            F_curr = 0.

            #for p in self.model.parameters():
            #    print(p)

            for batch_idx in list(loader_tr.batch_sampler):
            
                inputs, targets, _ = loader_tr.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                #scores = self.model(inputs)
                #print(self.criterion(scores, targets).item())

                '''l = [torch.flatten(p) for p in self.model.parameters()]
                flat = torch.cat(l)'''

                scores = torch.matmul(inputs,flat_orig[:-1].view(-1,1))

                scores = scores.view(-1) + flat_orig[-1]

                #print(scores.shape,targets.shape)

                error = scores - targets

                #F_curr += (self.criterion(scores, targets).item())*len(batch_idx) 
                F_curr += torch.sum(error*error)#sum_error(scores, targets).item()

            l = [torch.flatten(p) for p in self.model.parameters()]
            flat = torch.cat(l)
            l2_reg = torch.sum(flat*flat)

            constraint = torch.zeros(len(self.x_val_list),device=self.device)
            for j in range(len(self.x_val_list)):
                
                inputs_j, targets_j = self.x_val_list[j], self.y_val_list[j]
                scores_j = self.model(inputs_j)
                constraint[j] = self.criterion(scores_j, targets_j) - self.delta[j]

            multiplier = torch.dot(alphas,constraint)

            F_curr += (self.lam*l2_reg*len(curr_subset) + multiplier).item()


        main_optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr/10000)
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr/100)

        #ele_delta = self.delta.repeat(min(self.batch_size,self.y_trn[curr_subset].shape[0])).to(self.device)

        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

        rem_len = (len(curr_subset)-1)
        #print(rem_len)

        b_idxs = 0

        device_new = self.device #"cuda:2" #self.device #

        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, _ = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ele_delta = self.delta.to(self.device)#.repeat(targets.shape[0]).to(self.device)
            
            #print(flat_orig.view(1,-1))
            weights = flat_orig.view(1,-1).repeat(targets.shape[0], 1)
            #flat_orig.view(1,-1).repeat(targets.shape[0], 1)

            #print(weights[:2])
            #print(weights.shape)

            ele_alphas = alphas.detach().view(1,-1).repeat(targets.shape[0],1).to(self.device)
            #print(weights.dtype)#.shape)

            exp_avg_w = torch.zeros_like(weights)
            exp_avg_sq_w = torch.zeros_like(weights)

            exp_avg_a = torch.zeros_like(ele_alphas)
            exp_avg_sq_a = torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1)),dim=1)

            bias_correction1 = 1.0 
            bias_correction2 = 1.0 

            for i in range(p_epoch):

                fin_val_loss_g = torch.zeros_like(weights).to(device_new)
                #val_losses = torch.zeros_like(ele_delta).to(device_new)
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
                    
                    fin_val_loss_g += (torch.mean(val_loss_p[:,:,None]*exten_val[:,None,:],dim=0)*(ele_alphas[:,j][:,None]))

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val #mod_val,val_loss_g,
                    torch.cuda.empty_cache()

                #fin_val_loss_g /= len(loader_val.batch_sampler)
                #fin_val_loss_g = fin_val_loss_g.to(self.device)

                sum_fin_trn_loss_g = torch.zeros_like(weights).to(device_new)
                for batch_idx_trn in list(loader_tr.batch_sampler):
                    
                    inputs_trn, targets_trn,_ = loader_tr.dataset[batch_idx_trn]
                    inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)

                    exten_trn = torch.cat((inputs_trn,torch.ones(inputs_trn.shape[0]\
                        ,device=self.device).view(-1,1)),dim=1).to(device_new)
                    exten_trn_y = targets_trn.view(-1,1).repeat(1,targets.shape[0]).to(device_new)
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
                #sum_fin_trn_loss_g = sum_fin_trn_loss_g.to(self.device)

                trn_loss_g = 2*(torch.sum(exten_inp*weights,dim=1) - targets)
                fin_trn_loss_g = exten_inp*trn_loss_g[:,None]

                #print(fin_trn_loss_g[:2])
                #print(sum_fin_trn_loss_g[:2])

                fin_trn_loss_g = (sum_fin_trn_loss_g - fin_trn_loss_g)/rem_len

                #print(fin_trn_loss_g[:2])

                weight_grad = fin_trn_loss_g + 2*rem_len*self.lam*weights +fin_val_loss_g  #*ele_alphas[:,None]

                #print(weight_grad[:2])
                
                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = main_optimizer.param_groups[0]['lr'] * math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                alpha_grad = torch.zeros_like(ele_alphas).to(device_new)
                for j in range(len(self.x_val_list)):
                    
                    inputs_val, targets_val = self.x_val_list[j], self.y_val_list[j]
                    inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                    exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device)\
                        .view(-1,1)),dim=1).to(device_new)
                    #print(exten_val.shape)

                    exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,\
                        targets.shape[0])).to(device_new)
                    #print(exten_val_y[0])
                
                    val_loss_p = torch.matmul(exten_val,torch.transpose(weights, 0, 1).to(device_new))\
                         - exten_val_y #
                    
                    alpha_grad[:,j] = torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta[j]

                    del exten_val,exten_val_y,val_loss_p,inputs_val, targets_val
                    torch.cuda.empty_cache()

                #val_losses = val_losses.to(self.device)
                #alpha_grad = val_losses/len(loader_val.batch_sampler)-ele_delta

                exp_avg_a.mul_(beta1).add_(1.0 - beta1, alpha_grad)
                exp_avg_sq_a.mul_(beta2).addcmul_(1.0 - beta2, alpha_grad, alpha_grad)
                denom = exp_avg_sq_a.sqrt().add_(dual_optimizer.param_groups[0]['eps'])
                ele_alphas.addcdiv_(step_size, exp_avg_a, denom)

                #ele_alphas = ele_alphas + self.dual_lr*(alpha_grad)
                ele_alphas[ele_alphas < 0] = 0
                
            val_losses = torch.zeros(targets.shape[0],device=self.device)
            #ele_alphas = ele_alphas*((alpha_grad >0).float())
            for j in range(len(self.x_val_list)):
                    
                inputs_val, targets_val = self.x_val_list[j], self.y_val_list[j]
                inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                exten_val_y = targets_val.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
                
                val_loss = torch.matmul(exten_val,torch.transpose(weights, 0, 1)) - exten_val_y

                val_losses += (torch.mean(val_loss*val_loss,dim=0)-ele_delta[j])*ele_alphas[:,j]
                
            reg = torch.sum(weights*weights,dim=1)
            
            trn_losses = torch.zeros(targets.shape[0],device=self.device)
            for batch_idx_trn in list(loader_tr.batch_sampler):
                    
                inputs_trn, targets_trn,_ = loader_tr.dataset[batch_idx_trn]
                inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)

                exten_trn = torch.cat((inputs_trn,torch.ones(inputs_trn.shape[0],device=self.device).view(-1,1)),dim=1)
                #exten_trn = inputs_trn
                exten_trn_y = (targets_trn).view(-1,1).repeat(1,targets.shape[0]) #-weights[:,-1]
            
                trn_loss = torch.matmul(exten_trn,weights.T)- exten_trn_y #[:,:-1]

                trn_losses+= torch.sum(trn_loss*trn_loss,dim=0)

            trn_loss_ind = torch.sum(exten_inp*weights,dim=1) - targets

            trn_losses -= trn_loss_ind*trn_loss_ind
            
            abs_value = F_curr - (trn_losses + self.lam*reg*rem_len + val_losses)

            neg_ind = ((abs_value + 1e-4) < 0).nonzero().view(-1)
            
            abs_value [neg_ind] = torch.max(self.F_values)

            m_values[np.array(curr_subset[b_idxs*self.batch_size:(b_idxs+1)*self.batch_size])] = \
                abs_value

            b_idxs +=1


        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu().numpy())



