import numpy as np
import time
import torch
import math
import random
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, BatchSampler


class Glister_Linear_SetFunction_RModular_Regression(object):
    def __init__(self, trainloader, valloader, model,eta, device,selection_type, r=15):
        
        self.trainloader = trainloader
        self.valloader = valloader

        self.eta = eta  
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.r = r
        
        self.model = model
        self.eta = eta  
        
    def _compute_per_element_grads(self, theta_init):
        self.model.load_state_dict(theta_init)
        embDim = self.model.get_embedding_dim()

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if batch_idx == 0:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)
                   
                l0_grads = 2 * (out - targets)
                l1_grads = l0_grads.repeat(1, embDim) * l1

            else:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)

                l0_grads = 2 * (out - targets)
                batch_l1_grads = l0_grads.repeat(1, embDim) * l1

                l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

            torch.cuda.empty_cache()

            self.grads_per_elem = l1_grads

    def _update_grads_val(self, theta_init, grads_currX=None, first_init=False):
        self.model.zero_grad()
        embDim = self.model.get_embedding_dim()

        if first_init:
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:

                    with torch.no_grad():
                        out, l1 = self.model(inputs, last=True)

                    l0_grads = 2 * (out - targets)
                    l1_grads = l0_grads.repeat(1, embDim) * l1

                    self.init_out = out
                    self.init_l1 = l1
                    self.y_val = targets.view(out.shape[0], -1)

                else:
                    with torch.no_grad():
                        out, l1 = self.model(inputs, last=True)

                    l0_grads = 2 * (out - targets)
                    batch_l1_grads = l0_grads.repeat(1, embDim) * l1

                    l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y_val = torch.cat((self.y_val, targets.view(out.shape[0], -1)), dim=0)

        elif grads_currX is not None:
            with torch.no_grad():

                out_vec = self.init_out - (self.eta * self.init_out * grads_currX[0].view(1, -1))

                l0_grads = 2*(out_vec - self.y_val)
                l1_grads = l0_grads.repeat(1, embDim) * l1

        torch.cuda.empty_cache()
        self.grads_val_curr = torch.mean(l1_grads, dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val)
        return gains

    
    def _update_gradients_subset(self, grads_X, indices):
        
        grads_X += self.grads_per_elem[indices].sum(dim=0)

    # Same as before i.e full batch case! No use of dataloaders here!
    # Everything is abstracted away in eval call
    def  select(self, budget, model_params):

        #self.model.load_state_dict(model_params)

        self._compute_per_element_grads(model_params)
        self._update_grads_val(model_params,first_init=True)
        
        
        self.numSelected = 0
        greedySet = list()
        remainSet = list(range(self.N_trn))
        # RModular Greedy Selection Algorithm
        if self.selection_type == 'RGreedy':
            t_ng_start = time.time()  # naive greedy start time
            # subset_size = int((len(self.grads_per_elem) / r))
            selection_size = int(budget / self.r)
            while (self.numSelected < budget):
                # Try Using a List comprehension here!
                t_one_elem = time.time()
                rem_grads = self.grads_per_elem[remainSet]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                sorted_gains, indices = torch.sort(gains.view(-1), descending=True)
                selected_indices = [remainSet[index.item()] for index in indices[0:selection_size]]
                greedySet.extend(selected_indices)
                [remainSet.remove(idx) for idx in selected_indices]
                if self.numSelected == 0:
                    grads_currX = self.grads_per_elem[selected_indices].sum(dim=0).view(1, -1)
                else:  # If 1st selection, then just set it to bestId grads
                    self._update_gradients_subset(grads_currX, selected_indices)
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_currX)
                if self.numSelected % 1000 == 0:
                    # Printing bestGain and Selection time for 1 element.
                    print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
                self.numSelected += selection_size
            print("R greedy total time:", time.time() - t_ng_start)

        # Stochastic Greedy Selection Algorithm
        elif self.selection_type == 'Stochastic':
            t_ng_start = time.time()  # naive greedy start time
            subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
            while (self.numSelected < budget):
                # Try Using a List comprehension here!
                t_one_elem = time.time()
                subset_selected = random.sample(remainSet, k=subset_size)
                rem_grads = self.grads_per_elem[subset_selected]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                _, indices = torch.sort(gains.view(-1), descending=True)
                bestId = [subset_selected[indices[0].item()]]
                greedySet.append(bestId[0])
                remainSet.remove(bestId[0])
                self.numSelected += 1
                # Update info in grads_currX using element=bestId
                if self.numSelected > 1:
                    self._update_gradients_subset(grads_currX, bestId)
                else:  # If 1st selection, then just set it to bestId grads
                    grads_currX = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_currX)
                if (self.numSelected - 1) % 1000 == 0:
                    # Printing bestGain and Selection time for 1 element.
                    print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
            print("Stochastic Greedy total time:", time.time() - t_ng_start)

        elif self.selection_type == 'Naive':
            t_ng_start = time.time()  # naive greedy start time
            while (self.numSelected < budget):
                # Try Using a List comprehension here!
                t_one_elem = time.time()
                rem_grads = self.grads_per_elem[remainSet]
                gains = self.eval_taylor_modular(rem_grads)
                # Update the greedy set and remaining set
                # _, maxid = torch.max(gains, dim=0)
                _, indices = torch.sort(gains.view(-1), descending=True)
                bestId = [remainSet[indices[0].item()]]
                greedySet.append(bestId[0])
                remainSet.remove(bestId[0])
                self.numSelected += 1
                # Update info in grads_currX using element=bestId
                if self.numSelected == 1:
                    grads_currX = self.grads_per_elem[bestId[0]].view(1, -1)
                else:  # If 1st selection, then just set it to bestId grads
                    self._update_gradients_subset(grads_currX, bestId)
                # Update the grads_val_current using current greedySet grads
                self._update_grads_val(grads_currX)
                if (self.numSelected - 1) % 1000 == 0:
                    # Printing bestGain and Selection time for 1 element.
                    print("numSelected:", self.numSelected, "Time for 1:", time.time() - t_one_elem)
            print("Naive Greedy total time:", time.time() - t_ng_start)

        return list(greedySet)
