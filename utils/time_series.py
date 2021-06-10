import numpy as np
import pandas as pd
import os
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn import preprocessing

def process_time_series(datadir, past_length,col_name,name,save_data=True):

    x_trn = []
    y_trn = []

    path = os.path.join(datadir, 'prices-split-adjusted.csv')

    prices_split_adjusted = pd.read_csv(path,index_col='date', parse_dates=['date'])

    #Removing data with less than 1500 entries
    symbol = pd.DataFrame(prices_split_adjusted['symbol'].value_counts())
    omit = symbol[symbol['symbol'] < 1500].index
    prices_split_adjusted = prices_split_adjusted[~prices_split_adjusted['symbol'].isin(omit)]

    symbol = pd.DataFrame(prices_split_adjusted['symbol'].value_counts())

    for sym in symbol.index:

        stock = prices_split_adjusted[prices_split_adjusted['symbol'] == sym]

        data_len = len(stock.index)
        
        x_trn_slice = np.zeros((data_len-past_length,past_length))
        y_trn_slice = np.zeros(data_len-past_length)
        
        for ind in range(data_len-past_length):
            x_trn_slice[ind] = stock[col_name][stock.index[ind:ind+past_length]]
            y_trn_slice[ind] = stock[col_name][stock.index[ind+past_length]]

        x_trn.append(x_trn_slice)
        y_trn.append(y_trn_slice)

    finaL_x_trn = np.concatenate(x_trn, axis=0)
    finaL_y_trn = np.concatenate(y_trn, axis=0)

    
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = os.path.join(datadir, name+'_'+str(past_length)+'.data.npy') 
        target_np_path = os.path.join(datadir, name+'_'+str(past_length)+'.label.npy') 
        np.save(data_np_path, finaL_x_trn)
        np.save(target_np_path, finaL_y_trn)

    return finaL_x_trn, finaL_y_trn


def load_time_series_data (datadir, dset_name,past_length,clean=True):

    if os.path.isfile(os.path.join(datadir, dset_name+'_'+str(past_length)+'.data.npy')) and \
            os.path.isfile(os.path.join(datadir, dset_name+'_'+str(past_length)+'.label.npy')):
        
        x_trn = np.load(os.path.join(datadir, dset_name+'_'+str(past_length)+'.data.npy'))
        y_trn  = np.load(os.path.join(datadir, dset_name+'_'+str(past_length)+'.label.npy'))

    else:
    
        if dset_name == "NY_Stock_exchange_close":
            x_trn, y_trn  = process_time_series(datadir,past_length,'close',dset_name)

        elif dset_name == "NY_Stock_exchange_open":
            x_trn, y_trn  = process_time_series(datadir,past_length,'open',dset_name)

        elif dset_name == "NY_Stock_exchange_high":
            x_trn, y_trn  = process_time_series(datadir,past_length,'high',dset_name)

        elif dset_name == "NY_Stock_exchange_low":
            x_trn, y_trn  = process_time_series(datadir,past_length,'low',dset_name)

    y_trn = y_trn #+ 1600
    
    x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.005, random_state=42)
    
    if not clean:

        noise_size = int(len(y_trn) * 0.5)
        noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
        
        sigma = 40
        y_trn[noise_indices] = y_trn[noise_indices] + np.random.normal(0, sigma, noise_size)
    
    sc = MinMaxScaler() #StandardScaler()
    x_trn = sc.fit_transform(x_trn)
    x_val = sc.transform(x_val)
    x_tst = sc.transform(x_tst)

    '''sc_l = MinMaxScaler() #StandardScaler()
    y_trn = np.reshape(sc_l.fit_transform(np.reshape(y_trn,(-1,1))),(-1))'''
    #y_val = np.reshape(sc_l.fit_transform(np.reshape(y_val,(-1,1))),(-1))
    #y_tst = np.reshape(sc_l.fit_transform(np.reshape(y_tst,(-1,1))),(-1))

    fullset = (x_trn, y_trn)
    valset = (x_val, y_val)
    testset = (x_tst, y_tst)

    return fullset, valset, testset #, sc_l
