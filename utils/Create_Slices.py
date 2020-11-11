import numpy as np
import torch

def get_slices(data_name, data,labels,device):

    data_slices = []
    label_slices =[]

    if data_name == 'Community_Crime':
        protect_feature =[2,3,4,5]

        digit = np.zeros(data.shape[0],dtype=np.int8)
        #digit_tst = np.zeros(x_tst.shape[0],device=device)

        for i in protect_feature:

            low = np.min(data[:,i])
            high = np.max(data[:,i])
            bins = np.linspace(low, high, 5)
            digitized = np.digitize(data[:,i], bins)
            digit =  digit*10 + digitized

        classes = np.unique(digit)  
        
        for cl in classes:
            
            idx = (digit == cl).nonzero()[0].flatten()
            idx.tolist()
            data_slices.append(torch.from_numpy(data[idx]).float().to(device))
            label_slices.append(torch.from_numpy(labels[idx]).float().to(device))

    return data_slices, label_slices, classes

"""low = np.min(x_tst[:,i])
    high = np.max(x_tst[:,i])
    bins = np.linspace(low, high, 5)
    digitized = np.digitize(x_tst[:,i], bins)
    digit_tst =  digit_tst*10 + digitized"""


"""tst_classes = np.unique(digit_tst)  
for cl in tst_classes:

    idx = (digit_tst == cl).nonzero().flatten()
    idx.tolist()
    x_tst_list.append(torch.from_numpy(x_tst[idx]).float().to(device))
    y_tst_list.append(torch.from_numpy(y_tst[idx]).float().to(device))"""
