import math
import numpy as np
import time
import torch
from queue import PriorityQueue

from utils.custom_dataset import load_std_regress_data,CustomDataset,load_dataset_custom
from utils.Create_Slices import get_slices
from utils.time_series import load_time_series_data

import sys
import subprocess
import os

#Facility Location 
class SetFunctionFacLoc(object):

    def __init__(self, device, train_full_loader):#, valid_loader):
        
        self.train_loader = train_full_loader      
        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      #print(x)
      #print(x.shape)
      #print(y.shape)
      #print("n="+str(n)+" m="+str(m)+" d="+str(d))
      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)

      dist = torch.pow(x - y, exp).sum(2) 
      return dist 

    def compute_score(self):
      self.N = 0
      g_is = self.train_loader

      with torch.no_grad():
        '''for i, data_i in  enumerate(self.train_loader, 0):
          inputs_i, target_i = data_i
          inputs_i = inputs_i.to(self.device) #, target_i.to(self.device)
          self.N += inputs_i.size()[0]
          g_is.append(inputs_i)'''
        
        self.sim_mat = torch.zeros([self.N, self.N],dtype=torch.float32)

        first_i = True

        for i, g_i in enumerate(g_is, 0):

          if first_i:
            size_b = g_i.size(0)
            first_i = False

          for j, g_j in enumerate(g_is, 0):
            self.sim_mat[i*size_b: i*size_b + g_i.size(0), j*size_b: j*size_b + g_j.size(0)] = self.distance(g_i, g_j)
      self.const = torch.max(self.sim_mat).item()
      self.sim_mat = self.const - self.sim_mat
      #self.sim_mat = self.sim_mat.to(self.device)
      dist = self.sim_mat.sum(1)
      bestId = torch.argmax(dist).item()
      self.max_sim = self.sim_mat[bestId].to(self.device)
      return bestId


    def lazy_greedy_max(self, budget,logfile):
      
      starting = time.process_time() 
      id_first = self.compute_score()
      self.gains = PriorityQueue()
      for i in range(self.N):
        if i == id_first :
          continue
        curr_gain = (torch.max(self.max_sim ,self.sim_mat[i].to(self.device)) - self.max_sim).sum()
        self.gains.put((-curr_gain.item(),i))

      numSelected = 2
      second = self.gains.get()
      greedyList = [id_first, second[1]]
      self.max_sim = torch.max(self.max_sim,self.sim_mat[second[1]].to(self.device))

      ending = time.process_time()
      print("Kernel computation time ",ending-starting, file=logfile)

      starting = time.process_time()
      while(numSelected < budget):

          if self.gains.empty():
            break

          elif self.gains.qsize() == 1:
            bestId = self.gains.get()[1]

          else:
   
            bestGain = -np.inf
            bestId = None
            
            while True:

              first =  self.gains.get()

              if bestId == first[1]: 
                break

              curr_gain = (torch.max(self.max_sim, self.sim_mat[first[1]].to(self.device)) - self.max_sim).sum()
              self.gains.put((-curr_gain.item(), first[1]))


              if curr_gain.item() >= bestGain:
                  
                bestGain = curr_gain.item()
                bestId = first[1]

          greedyList.append(bestId)
          numSelected += 1

          self.max_sim = torch.max(self.max_sim,self.sim_mat[bestId].to(self.device))

      #print()
      #gamma = self.compute_gamma(greedyList)

      ending = time.process_time()
      print("Selectio time ",ending-starting, file=logfile)

      return greedyList

def run_stochastic_Facloc( data, targets, sub_budget, budget,logfile,device='cpu'):
    
    #model = model.to(device)
    '''approximate_error = 0.01
    #per_iter_bud = 10
    sample_size = int(len(data) / num_iterations * math.log(1 / approximate_error))'''
    #greedy_batch_size = 1200

    num_iterations = int(math.ceil(len(data)/sub_budget))
    trn_indices = list(np.arange(len(data)))
    facloc_indices = []

    per_iter_bud = int(budget/num_iterations)

    trn_batch = 1200
 
    for i in range(num_iterations):
        
        rem_indices = list(set(trn_indices).difference(set(facloc_indices)))

        state = np.random.get_state()
        np.random.seed(i*i)
        sub_indices = np.random.choice(rem_indices, size=sub_budget, replace=False)
        np.random.set_state(state)

        data_subset = data[sub_indices].cpu()
        targets_subset = targets[sub_indices].cpu()
        train_loader_greedy = []
        for item in range(math.ceil(sub_budget /trn_batch)):
          inputs = data_subset[item*trn_batch:(item+1)*trn_batch]
          target  = targets_subset[item*trn_batch:(item+1)*trn_batch]
          train_loader_greedy.append((inputs,target))
        
        #train_loader_greedy.append((data_subset, targets_subset))
        setf_model = SetFunctionFacLoc(device, train_loader_greedy)
        idxs = setf_model.lazy_greedy_max(min(per_iter_bud,budget-len(facloc_indices)),logfile)#, model)
        facloc_indices.extend([sub_indices[idx] for idx in idxs])
    return facloc_indices

## Convert to this argparse
datadir = sys.argv[1]
data_name = sys.argv[2]
frac = float(sys.argv[3])
is_time = bool(int(sys.argv[4]))
if is_time:
    past_length = int(sys.argv[5])

device = "cuda" if torch.cuda.is_available() else "cpu"

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

budget = frac*len(y_trn) #[int(float(i)*len(y_trn)) for i in frac]

all_logs_dir = './results/FacLoc/' + data_name
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')

index =run_stochastic_Facloc(x_trn, y_trn, 10000, budget,logfile,device=device)

logfile.writelines(index)