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


## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None):       
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = torch.from_numpy(data.astype('float32'))#.to(device)
            self.targets = torch.from_numpy(target)#.to(device)
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



def german_load(path,dim, save_data=False):

    data = []
    target = []

    indices = [0,2,3,5,6,8,9,11,13,14,16,18,19]
    values = [4,4,10,5,5,5,3,4,3,3,4,2,2]

    list_dict = []
    for i in range(len(indices)):
        enum=enumerate(['A'+str(indices[i]+1)+str(j) for j in range(values[i]+1)])
        list_dict.append( dict((j,i) for i,j in enum))

    with open(path) as fp:
        line = fp.readline()
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(" ")]

            target.append(float(temp[-1])-1)
            
            temp_data = [0.0]*dim
            
            #print(temp)

            for i in range(len(temp[:-1])):

                if i in indices:
                    ind = indices.index(i)
                    temp_data[i] =  list_dict[ind][temp[i]]
                else:
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

def community_online_news(path,dim, save_data=False):

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")][2:]

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

'''def banking_load(path,dim=14, save_data=False):
    
    enum=enumerate(["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",\
        "blue-collar","self-employed","retired","technician","services"])
    job = dict((j,i) for i,j in enum)

    enum=enumerate(["married","divorced","single"])
    marital_status = dict((j,i) for i,j in enum)

    enum=enumerate(["unknown","secondary","primary","tertiary"])
    education = dict((j,i) for i,j in enum)

    enum=enumerate(["yes","no"])
    yes_no = dict((j,i) for i,j in enum)

    enum=enumerate(["unknown","other","failure","success"])
    pout = dict((j,i) for i,j in enum)

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")]

            if '?' in temp  or len(temp) == 1:
                line = fp.readline()
                continue

            if temp[-1].strip() == "no":
                target.append(0) 
            else:
                target.append(1)
            
            temp_data = [0]*dim
            count = 0
            #print(temp)

            for i in temp[:-1]:

                if count == 1:
                    temp_data[count] =  job[i.strip()]
                elif count == 2:
                    temp_data[count] =  marital_status[i.strip()]
                elif count == 3:
                    temp_data[count] =  education[i.strip()]
                elif count in [4,6]:
                    temp_data[count] =  yes_no[i.strip()]

                elif count == 7:
                    temp_data[count] =  relationship[i.strip()]
                elif count == 8:
                    temp_data[count] =  race[i.strip()]
                elif count == 9:
                    temp_data[count] =  sex[i.strip()]
                elif count == 13:
                    temp_data[count] =  native_country[i.strip()]
                else:
                    temp_data[count] = float(i)
                temp_data[count] = float(temp_data[count])
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
    return (X_data, Y_label)'''


def census_load(path,dim, save_data=False):
    
    enum=enumerate(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 
        'Never-worked'])
    workclass = dict((j,i) for i,j in enum)

    enum=enumerate(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', 
        '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    education = dict((j,i) for i,j in enum)

    enum=enumerate(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    marital_status = dict((j,i) for i,j in enum)

    enum=enumerate(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    occupation = dict((j,i) for i,j in enum)

    enum=enumerate(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    relationship = dict((j,i) for i,j in enum)

    enum=enumerate(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    race = dict((j,i) for i,j in enum)

    sex ={'Female':0,'Male':1}

    enum=enumerate(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
     'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
     'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 
     'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 
     'Holand-Netherlands'])
    native_country = dict((j,i) for i,j in enum)

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")]

            if '?' in temp  or len(temp) == 1:
                line = fp.readline()
                continue

            if temp[-1].strip() == "<=50K" or temp[-1].strip() == "<=50K.":
                target.append(0) 
            else:
                target.append(1)
            
            temp_data = [0]*dim
            count = 0
            #print(temp)

            for i in temp[:-1]:

                if count == 1:
                    temp_data[count] =  workclass[i.strip()]
                elif count == 3:
                    temp_data[count] =  education[i.strip()]
                elif count == 5:
                    temp_data[count] =  marital_status[i.strip()]
                elif count == 6:
                    temp_data[count] =  occupation[i.strip()]
                elif count == 7:
                    temp_data[count] =  relationship[i.strip()]
                elif count == 8:
                    temp_data[count] =  race[i.strip()]
                elif count == 9:
                    temp_data[count] =  sex[i.strip()]
                elif count == 13:
                    temp_data[count] =  native_country[i.strip()]
                else:
                    temp_data[count] = float(i)
                temp_data[count] = float(temp_data[count])
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
    
def load_dataset_custom (datadir, dset_name,isnumpy=True):

    if dset_name == "census":
        trn_file = os.path.join(datadir, 'adult.data')
        tst_file = os.path.join(datadir, 'adult.test')

        data_dims = 14
        num_cls = 2

        x_trn, y_trn = census_load(trn_file, dim=data_dims)
        x_tst, y_tst = census_load(tst_file, dim=data_dims)
        
        '''x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)'''

        x_trn = np.concatenate((x_trn, x_tst), axis=0)
        y_trn = np.concatenate((y_trn, y_tst), axis=0)

        if isnumpy:
            fullset = (x_trn, y_trn)
            #valset = (x_val, y_val)
            #testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            #valset = CustomDataset(x_val, y_val)
            #testset = CustomDataset(x_tst, y_tst)

        return fullset, data_dims #, valset, testset, 

    elif dset_name == "Community_Crime":
        '''trn_file = os.path.join(datadir, 'communities.data')

        data_dims = 122

        x_trn, y_trn = community_crime_load(trn_file, dim=data_dims)'''

        #x_trn = preprocessing.normalize(x_trn)
        
        '''x_trn, x_tst, y_trn, y_tst= train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)'''

        x_trn, y_trn = clean_communities_full(os.path.join(datadir, 'communities.csv'))

        #print(x_trn.shape[1])

        if isnumpy:
            fullset = (x_trn, y_trn)
            #valset = (x_val, y_val)
            #testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            #valset = CustomDataset(x_val, y_val)
            #testset = CustomDataset(x_tst, y_tst)

        return fullset, x_trn.shape[1] # valset, testset,

    elif dset_name == 'OnlineNewsPopularity':
        trn_file = os.path.join(datadir, 'OnlineNewsPopularity.csv')

        data_dims = 58

        x_trn, y_trn = community_online_news(trn_file, dim=data_dims)
        
        '''x_trn, x_tst, y_trn, y_tst= train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)'''

        if isnumpy:
            fullset = (x_trn, y_trn)
            
        else:
            fullset = CustomDataset(x_trn, y_trn)

        return fullset, data_dims  # valset, testset,

    elif dset_name == 'LawSchool':

        x_trn, y_trn = clean_lawschool_full(os.path.join(datadir, 'lawschool.csv'))

        if isnumpy:
            fullset = (x_trn, y_trn)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            
        return fullset, x_trn.shape[1] 

    elif dset_name == 'German_credit':

        trn_file = os.path.join(datadir, 'german.data')

        data_dims = 20

        x_trn, y_trn = german_load(trn_file, dim=data_dims)

        if isnumpy:
            fullset = (x_trn, y_trn)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            
        return fullset, data_dims

def load_std_regress_data (datadir, dset_name,isnumpy=True):

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

    elif dset_name == "census":
        trn_file = os.path.join(datadir, 'adult.data')
        tst_file = os.path.join(datadir, 'adult.test')

        data_dims = 14

        x_trn, y_trn = census_load(trn_file, dim=data_dims)
        x_tst, y_tst = census_load(tst_file, dim=data_dims)

        x_trn = np.concatenate((x_trn, x_tst), axis=0)
        y_trn = np.concatenate((y_trn, y_tst), axis=0)

    elif dset_name == "Community_Crime":

        x_trn, y_trn = clean_communities_full(os.path.join(datadir, 'communities.csv'))

    elif dset_name == 'LawSchool':

        x_trn, y_trn = clean_lawschool_full(os.path.join(datadir, 'lawschool.csv'))

    elif dset_name == 'OnlineNewsPopularity':
        trn_file = os.path.join(datadir, 'OnlineNewsPopularity.csv')

        data_dims = 58

        x_trn, y_trn = community_online_news(trn_file, dim=data_dims)

    elif dset_name == "synthetic":

        data_dims = 30 #100
        samples = 5000#1000000

        np.random.seed(42)
        
        #avg = np.random.rand(data_dims)*5
        #cov = np.diag(np.random.randint(50, size=data_dims))

        avg = np.zeros(data_dims)
        cov = np.identity(data_dims)

        x_trn = np.random.multivariate_normal(avg, cov, samples)#.T
        #print(x_trn.shape)

        #w = np.random.normal(np.mean(avg), np.random.randint(50, size=1)[0], data_dims)
        w = np.random.normal(0,10, data_dims) #0.25

        sigma = 30#np.random.randint(50, size=1)[0]/50
        #print(sigma)
        #print(w)
        
        y_trn = x_trn.dot(w) + np.random.normal(0, sigma, samples)
        #print(y_trn[:20])
        #print(x_trn.dot(w)[:20])
        #print((y_trn- x_trn.dot(w))[:20])

    '''elif dset_name == "MSD":
        trn_file = os.path.join(datadir, 'ablone_scle.txt')
        x_trn, y_trn  = libsvm_file_load(trn_file,8)'''

    if dset_name == "MSD":
        tst_file = os.path.join(datadir, 'YearPredictionMSD.t')
        x_tst, y_tst  = libsvm_file_load(tst_file,90)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.003, random_state=42)
    else:
        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    if dset_name not in ["census"]:
    
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

    



