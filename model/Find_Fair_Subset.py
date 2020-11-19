import math
import numpy as np
import time
import torch
import torch.nn as nn
import copy


class FindSubset(object):
    def __init__(self, x_trn, y_trn, x_val_list, y_val_list,model,loss,val_size,device,delta,lr):
        
        self.x_trn = x_trn
        self.y_trn = y_trn
        #self.trn_batch = trn_batch

        self.x_val_list = x_val_list
        self.y_val_list = y_val_list

        self.model = model
        self.criterion = loss 
        self.device = device

        self.delta = delta
        self.lr = lr
        self.val_size = val_size

    def precompute(self,f_pi_epoch,p_epoch):

        self.F_phi = 0

        #self.model.load_state_dict(theta_init)

        alphas = torch.ones_like(self.val_size,requires_grad=True)
        
        main_optimizer = torch.optim.Adam([{'params': alphas,'lr':self.lr*10}], lr=self.lr)
        #{'params': self.model.parameters()},

        alphas.requires_grad = False

        for i in range(f_pi_epoch):
            
            alpha_extend = torch.repeat_interleave(alphas,val_size, dim=0)



            val_scores = self.model(self.x_val_list)
            constraint = self.criterion(val_scores, self.y_val_list) - self.delta
            multiplier = torch.dot(alpha_extend,constraint)

            loss = criterion(scores, targets) +  reg_lambda*l2_reg*len(idxs) + multiplier
            loss.backward()
            main_optimizer.step()

            if i % print_every == 0:  # Print Training and Validation Loss
                print('Epoch:', i + 1, 'SubsetTrn', loss.item())
            
            for param in self.main_model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True

            #print(main_optimizer.param_groups)

            val_scores = self.model(self.x_val_list)
            constraint = self.criterion(val_scores, self.y_val_list) - self.delta
            multiplier = torch.dot(alpha_extend,-1.0*constraint)
            
            main_optimizer.zero_grad()
            multiplier.backward()
            main_optimizer.step()

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)

            for param in main_model.parameters():
                param.requires_grad = True



    #def return_subset(self,curr_subset,):
