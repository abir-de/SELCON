import math
import numpy as np
import time
import torch
import torch.nn as nn
import copy


class FindSubset(object):
    def __init__(self, x_trn, y_trn, x_val, y_val,model,optimizer,loss,device,delta,lr,lam):
        
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
        self.optimizer = optimizer

    def precompute(self,f_pi_epoch,p_epoch):

        #self.model.load_state_dict(theta_init)

        print("starting Pre compute")
        alphas = torch.rand_like(self.delta,requires_grad=True) 

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,40,100],\
        #    gamma=0.5) #[e*2 for e in change]

        alphas.requires_grad = False

        #Compute F_phi
        for i in range(f_pi_epoch):
            
            self.optimizer.zero_grad()
            l2_reg = 0

            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            scores_val = self.model(self.x_val)
            constraint = self.criterion(scores_val, self.y_val) - self.delta
            multiplier = alphas*constraint #torch.dot(alphas,constraint)

            loss = multiplier
            self.F_phi = loss.item()
            loss.backward()
            
            self.optimizer.step()
            #scheduler.step()
            
            for param in self.model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True

            scores_val = self.model(self.x_val)
            constraint = self.criterion(scores_val, self.y_val) - self.delta
            multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
            
            multiplier.backward()
            
            self.optimizer.step()

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            #print(alphas)

            for param in self.model.parameters():
                param.requires_grad = True

        print("Finishing F phi")

        self.F_values = torch.zeros(len(self.x_trn))
        alphas_orig = copy.deepcopy(alphas)
        cached_state_dict = copy.deepcopy(self.model.state_dict())

        for trn_id in range(len(self.x_trn)):
            alphas = copy.deepcopy(alphas_orig)
            self.model.load_state_dict(cached_state_dict)

            for i in range(p_epoch):
                
                inputs, targets = self.x_trn[trn_id], self.y_trn[trn_id].view(-1)
                self.optimizer.zero_grad()
                l2_reg = 0
                
                scores = self.model(inputs)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                
                scores_val = self.model(self.x_val)
                constraint = self.criterion(scores_val, self.y_val) - self.delta
                multiplier = alphas*constraint #torch.dot(alphas,constraint)

                #print(scores,targets)

                loss = self.criterion(scores, targets) +  self.lam*l2_reg + multiplier
                self.F_values[trn_id] = loss.item()
                loss.backward()
                
                self.optimizer.step()
                #scheduler.step()

                for param in self.model.parameters():
                    param.requires_grad = False
                alphas.requires_grad = True

                scores_val = self.model(self.x_val)
                constraint = self.criterion(scores_val, self.y_val) - self.delta
                multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
                
                multiplier.backward()
                
                self.optimizer.step()

                alphas.requires_grad = False
                alphas.clamp_(min=0.0)
                #print(alphas)

                for param in self.model.parameters():
                    param.requires_grad = True

        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,alphas,budget):

        m_values = copy.deepcopy(self.F_values) #torch.zeros(len(self.x_trn))
        
        self.model.load_state_dict(theta_init)

        with torch.no_grad():

            inputs, targets = self.x_trn[curr_subset], self.y_trn[curr_subset]
            self.optimizer.zero_grad()
            l2_reg = 0
            
            scores = self.model(inputs)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            scores_val = self.model(self.x_val)
            constraint = self.criterion(scores_val, self.y_val) - self.delta
            multiplier = alphas*constraint #torch.dot(alphas,constraint)

            F_curr = (self.criterion(scores, targets) +  self.lam*l2_reg + multiplier).item()

        alphas_orig = copy.deepcopy(alphas)
        cached_state_dict = copy.deepcopy(self.model.state_dict())

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,40,100],\
        #    gamma=0.5) #[e*2 for e in change]

        alphas.requires_grad = False
        
        for sub_id in curr_subset:
            removed = copy.deepcopy(curr_subset)
            removed.remove(sub_id)

            alphas = copy.deepcopy(alphas_orig)
            self.model.load_state_dict(cached_state_dict)

            for i in range(p_epoch):
                
                inputs, targets = self.x_trn[removed], self.y_trn[removed]
                self.optimizer.zero_grad()
                l2_reg = 0
                
                scores = self.model(inputs)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                
                scores_val = self.model(self.x_val)
                constraint = self.criterion(scores_val, self.y_val) - self.delta
                multiplier = alphas*constraint #torch.dot(alphas,constraint)

                loss = self.criterion(scores, targets) +  self.lam*l2_reg + multiplier
                m_values[sub_id] = F_curr - loss.item()
                loss.backward()
                
                self.optimizer.step()
                #scheduler.step()

                for param in self.model.parameters():
                    param.requires_grad = False
                alphas.requires_grad = True

                scores_val = self.model(self.x_val)
                constraint = self.criterion(scores_val, self.y_val) - self.delta
                multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
                
                multiplier.backward()
                
                self.optimizer.step()

                alphas.requires_grad = False
                alphas.clamp_(min=0.0)
                #print(alphas)

                for param in self.model.parameters():
                    param.requires_grad = True

        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu())


        