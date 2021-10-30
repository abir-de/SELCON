import copy
import datetime
import numpy as np
import os
import subprocess
import sys
import time
import torch

from utils.custom_dataset import load_std_regress_data, load_dataset_custom
from utils.Create_Slices import get_slices

from utils.time_series import load_time_series_data

torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Using Device:", device)
       
def load_def_data(data_name, datadir = '.\Dataset', is_time = False, past_length = 100):
    '''Loads the default datasets used to present the results
    in the paper.
    
    Arguments:
        data_dir: Directory in which the default datasets are stored
        data_name: Name of dataset to be used for experiment
        is_time: 1 if dataset is time series; 0 otherwise
        past_length: 
        
    Returns:
        Tuple of numpy arrays: (x_trn, y_train), (x_val, y_val), (x_tst, y_tst)
    '''
    if is_time:
        fullset, valset, testset = load_time_series_data(datadir, data_name, past_length) #, sc_trans

        x_trn,y_trn =  torch.from_numpy(fullset[0]).float(),torch.from_numpy(fullset[1]).float()
        x_val,y_val =  torch.from_numpy(valset[0]).float(),torch.from_numpy(valset[1]).float()
        x_tst, y_tst = torch.from_numpy(testset[0]).float(),torch.from_numpy(testset[1]).float()

    elif data_name in ['Community_Crime','census','LawSchool']:
        
        datadir = datadir + '/' + data_name + '/'

        fullset, data_dims = load_dataset_custom(datadir, data_name, True)

        if data_name == 'Community_Crime':
            x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
                = get_slices(data_name, fullset[0], fullset[1], device, 3)

            change = [20,40,80,160]

        elif data_name == 'census':
            x_trn, y_trn, x_val_list, y_val_list, val_classes,x_tst_list, y_tst_list, tst_classes\
                = get_slices(data_name, fullset[0], fullset[1], device)

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
        datadir = datadir + '/' + data_name + '/'        
        
        fullset, valset, testset = load_std_regress_data (datadir, data_name, True)

        x_trn,y_trn =  torch.from_numpy(fullset[0]).float(),torch.from_numpy(fullset[1]).float()
        x_val,y_val =  torch.from_numpy(valset[0]).float(),torch.from_numpy(valset[1]).float()
        x_tst, y_tst = torch.from_numpy(testset[0]).float(),torch.from_numpy(testset[1]).float()

    return (x_trn, y_trn), (x_val, y_val), (x_tst, y_tst)

def get_data(x_train, x_val, y_train, y_val):

    x_trn,y_trn =  torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float()
    x_val,y_val =  torch.from_numpy(x_val).float(),torch.from_numpy(y_val).float()
    
    return x_trn, x_val, y_trn, y_val
        

