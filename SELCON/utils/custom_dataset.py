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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None):       
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = torch.from_numpy(data.astype('float32')).to(device)
            self.targets = torch.from_numpy(target).to(device)
        else:
            self.data = data#.astype('float32')
            self.targets = target

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label) #.astype('float32')

class CustomDataset_WithId(Dataset):
    def __init__(self, data, target, transform=None):       
        self.transform = transform
        self.data = data #.astype('float32')
        self.targets = target
        self.X = self.data
        self.Y = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return sample_data, label,idx #.astype('float32')

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                # convert float NaN --> string NaN
                output[col] = output[col].fillna('NaN')
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

## Utility function to load datasets from libsvm datasets
def csv_file_load(path,dim,skip=False,save_data=False):
    data = []
    target = []
    with open(path) as fp:
       if skip:
           line = fp.readline()
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(",")]
        target.append(int(float(temp[-1]))) # Class Number. # Not assumed to be in (0, K-1)
        temp_data = [0]*dim
        count = 0
        for i in temp[:-1]:
            #ind, val = i.split(':')
            temp_data[count] = float(i)
            count += 1
        data.append(temp_data)
        line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)

def libsvm_file_load(path,dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(" ")]
        target.append(int(float(temp[0]))) # Class Number. # Not assumed to be in (0, K-1)
        temp_data = [0]*dim
        
        for i in temp[1:]:
            if len(i) > 1: 
                ind,val = i.split(':')
                temp_data[int(ind)-1] = float(val)
        data.append(temp_data)
        line = fp.readline()
    X_data = np.array(data,dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)

def clean_lawschool_full(path):
   
    df = pd.read_csv(path)
    df = df.dropna()
    # remove y from df
    y = df['ugpa']
    y = y / 4
    df = df.drop('ugpa', 1)
    # convert gender variables to 0,1
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    # add bar1 back to the feature set
    df_bar = df['bar1']
    df = df.drop('bar1', 1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    #df['race'] = [int(race == 7.0) for race in df['race']]
    #a = df['race']
    return df.to_numpy(), y.to_numpy()

def majority_pop(a):
    """
    Identify the main ethnicity group of each community
    """
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    races = [B, W, A, H]
    maj = a.apply(pd.Series.idxmax, axis=1)
    return maj

def clean_communities_full(path):
    """
    Extract black and white dominant communities; 
    sub_size : number of communities for each group
    """
    df = pd.read_csv(path)
    df = df.fillna(0)
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    sens_features = [2, 3, 4, 5]
    df_sens = df.iloc[:, sens_features]

    # creating labels using crime rate
    Y = df['ViolentCrimesPerPop']
    df = df.drop('ViolentCrimesPerPop', 1)

    maj = majority_pop(df_sens)

    # remap the values of maj
    a = maj.map({B : 0, W : 1, A : 2, H : 3})
   
    df['race'] = a
    df = df.drop(H, 1)
    df = df.drop(B, 1)
    df = df.drop(W, 1)
    df = df.drop(A, 1)

    #print(df.head())

    return df.to_numpy(), Y.to_numpy()

def house_price_load(trn_path,tst_path,save_data=False):

    train_csv = pd.read_csv(trn_path)
    test_csv = pd.read_csv(tst_path)

    drop_columns = (train_csv.isnull().sum().sort_values(ascending=False).\
        loc[lambda x : x> .90*1460]).index.to_list()

    train_clean = train_csv.drop(drop_columns, axis = 'columns', errors = 'ignore')
    test_clean = test_csv.drop(drop_columns, axis = 'columns', errors = 'ignore')

    train_10_percent_missing_features = train_clean.isnull().sum().sort_values(ascending=False).loc[lambda x : (x<.10*1460)  & (x != 0)].index.to_list()
    train_10_percent_missing_features_cat = train_clean[train_10_percent_missing_features].select_dtypes('object').columns.to_list()
    train_10_percent_missing_features_num = train_clean[train_10_percent_missing_features].select_dtypes('number').columns.to_list()

    train_clean[train_10_percent_missing_features_cat] = train_clean[train_10_percent_missing_features_cat].fillna(train_clean[train_10_percent_missing_features_cat].mode().iloc[0])
    train_clean[train_10_percent_missing_features_num] = train_clean[train_10_percent_missing_features_num].fillna(train_clean[train_10_percent_missing_features_num].median().iloc[0])

    test_10_percent_missing_features = test_clean.isnull().sum().sort_values(ascending=False).loc[lambda x : (x<.10*1460)  & (x != 0)].index.to_list()
    test_10_percent_missing_features_cat = test_clean[test_10_percent_missing_features].select_dtypes('object').columns.to_list()
    test_10_percent_missing_features_num = test_clean[test_10_percent_missing_features].select_dtypes('number').columns.to_list()

    test_clean[test_10_percent_missing_features_cat] = test_clean[test_10_percent_missing_features_cat].fillna(test_clean[test_10_percent_missing_features_cat].mode().iloc[0])
    test_clean[test_10_percent_missing_features_num] = test_clean[test_10_percent_missing_features_num].fillna(test_clean[test_10_percent_missing_features_num].median().iloc[0])

    train_clean["LotFrontage"] = train_clean["LotFrontage"].fillna(train_clean["LotFrontage"].median())
    test_clean["LotFrontage"] = test_clean["LotFrontage"].fillna(test_clean["LotFrontage"].median())

    train_clean.drop('Id', axis = 1, inplace = True)
    test_clean.drop('Id', axis = 1, inplace = True)

    trn_y = train_clean['SalePrice'].to_frame()
    #tst_y = test_clean['SalePrice'].to_frame()

    train_clean.drop(['SalePrice'], axis = 1, inplace = True)
    #test_clean.drop(['SalePrice'], axis = 1, inplace = True)
 
    X_train_clean_index = train_clean.index.to_list()
    X_total = train_clean.append(test_clean,ignore_index = True)
    X_test_clean_index = np.setdiff1d(X_total.index.to_list() ,X_train_clean_index)

    cat_features = X_total.select_dtypes(['object']).columns.to_list()
    X_total_encoded = MultiColumnLabelEncoder(columns = cat_features).fit_transform(X_total)
    
    X_train_clean_encoded = X_total_encoded.iloc[X_train_clean_index, :]
    X_test_clean_encoded = X_total_encoded.iloc[X_test_clean_index, :].reset_index(drop = True) 
    
    return X_train_clean_encoded.to_numpy(),trn_y.to_numpy()#,X_test_clean_encoded,tst_y

def community_crime_load(path,dim, save_data=False):

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")][5:]

            target.append(float(temp[-1]))
            
            temp_data = [0.0]*dim
            
            #print(temp)

            for i in range(len(temp[:-1])):

                if temp[i] != '?':
                    temp_data[i] = float(temp[i])
            
            data.append(temp_data)
            line = fp.readline()
    
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


def load_dataset_custom (datadir, dset_name,isnumpy=True):

    if dset_name == "Community_Crime":

        x_trn, y_trn = clean_communities_full(os.path.join(datadir, 'communities.csv'))

        if isnumpy:
            fullset = (x_trn, y_trn)
            
        else:
            fullset = CustomDataset(x_trn, y_trn)

        return fullset, x_trn.shape[1] 

    elif dset_name == 'LawSchool':

        x_trn, y_trn = clean_lawschool_full(os.path.join(datadir, 'lawschool.csv'))

        if isnumpy:
            fullset = (x_trn, y_trn)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            
        return fullset, x_trn.shape[1] 

def load_std_regress_data (datadir, dset_name,isnumpy=True,clean=True):

    if dset_name == "cadata":
        trn_file = os.path.join(datadir, 'cadata.txt')
        x_trn, y_trn  = libsvm_file_load(trn_file,8)
    
    elif dset_name == "abalone":
        trn_file = os.path.join(datadir, 'abalone_scale.txt')
        x_trn, y_trn  = libsvm_file_load(trn_file,8)

    elif dset_name == "cpusmall":
        trn_file = os.path.join(datadir, 'cpusmall_scale.txt')
        x_trn, y_trn  = libsvm_file_load(trn_file,12)

    elif dset_name == "housing":
        trn_file = os.path.join(datadir, 'housing_scale.txt')
        x_trn, y_trn  = libsvm_file_load(trn_file,13)

    elif dset_name == "mg":
        trn_file = os.path.join(datadir, 'mg_scale.txt')
        x_trn, y_trn  = libsvm_file_load(trn_file,6)

    elif dset_name == "MSD":
        trn_file = os.path.join(datadir, 'YearPredictionMSD')
        x_trn, y_trn  = libsvm_file_load(trn_file,90)

    elif dset_name == "house_pricing":
        trn_file = os.path.join(datadir, 'train.csv')
        test_file = os.path.join(datadir, 'test.csv')
        x_trn, y_trn  = house_price_load(trn_file,test_file) #,x_tst,y_tst

    elif dset_name == "synthetic":

        data_dims = 30 #100
        samples = 5000#1000000

        np.random.seed(42)

        avg = np.zeros(data_dims)
        cov = np.identity(data_dims)

        x_trn = np.random.multivariate_normal(avg, cov, samples)#.T

        w = np.random.normal(0,10, data_dims) #0.25

        sigma = 30

        y_trn = x_trn.dot(w) + np.random.normal(0, sigma, samples)
    
    #elif dset_name == "house_pricing":
    #    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    
    if dset_name == "MSD":
        tst_file = os.path.join(datadir, 'YearPredictionMSD.t')
        x_tst, y_tst  = libsvm_file_load(tst_file,90)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.005, random_state=42)
    else:
        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    if not clean:

        noise_size = int(len(y_trn) * 0.5)
        noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
        
        sigma = 40
        y_trn[noise_indices] = y_trn[noise_indices] + np.random.normal(0, sigma, noise_size)

    sc = StandardScaler()
    x_trn = sc.fit_transform(x_trn)
    x_val = sc.transform(x_val)
    x_tst = sc.transform(x_tst)

    sc_l = StandardScaler()
    y_trn = np.reshape(sc_l.fit_transform(np.reshape(y_trn,(-1,1))),(-1))
    y_val = np.reshape(sc_l.fit_transform(np.reshape(y_val,(-1,1))),(-1))
    y_tst = np.reshape(sc_l.fit_transform(np.reshape(y_tst,(-1,1))),(-1))

    if isnumpy:
        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)

    else:
        fullset = CustomDataset(x_trn, y_trn)
        valset = CustomDataset(x_val, y_val)
        testset = CustomDataset(x_tst, y_tst)

    return fullset, valset, testset

    



