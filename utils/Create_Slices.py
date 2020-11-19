import numpy as np
import torch

def gen_rand_prior_indices(curr_size,num_cls,x_trn,y_trn,remainList=None):

    per_sample_budget = int(curr_size/num_cls)
    if remainList is None:
        per_sample_count = [len(torch.where(y_trn == x)[0]) for x in np.arange(num_cls)]
        total_set = list(np.arange(N))
    else:
        per_sample_count = [len(torch.where(y_trn[remainList] == x)[0]) for x in np.arange(num_cls)]
        total_set = remainList
    indices = []
    count = 0
    for i in range(num_cls):
        if remainList is None:
            label_idxs = torch.where(y_trn == i)[0].cpu().numpy()
        else:
            label_idxs = torch.where(y_trn[remainList] == i)[0].cpu().numpy()
            label_idxs = np.array(remainList)[label_idxs]

        if per_sample_count[i] > per_sample_budget:
            indices.extend(list(np.random.choice(label_idxs, size=per_sample_budget, replace=False)))
        else:
            indices.extend(label_idxs)
            count += (per_sample_budget - per_sample_count[i])
    for i in indices:
        total_set.remove(i)
    indices.extend(list(np.random.choice(total_set, size= count, replace=False)))
    return indices

def get_slices(data_name, data,labels,device,buckets=None):

    #data_slices = []
    #abel_slices =[]
    
    val_data_slices = []
    val_label_slices =[]

    tst_data_slices = []
    tst_label_slices =[]

    if data_name == 'Community_Crime':
        protect_feature =[2,3,4,5]

        data_class =[]

        N = int(0.1*len(data)/(buckets*len(protect_feature)))

        total_set = set(list(np.arange(len(data))))
        
        for i in protect_feature:            

            digit = np.ones(data.shape[0],dtype=np.int8)*(i-1)
            low = np.min(data[:,i])
            high = np.max(data[:,i])
            bins = np.linspace(low, high, buckets)
            digitized = np.digitize(data[:,i], bins)
            digit =  digit*10 + digitized

            classes,times = np.unique(digit,return_counts=True) 
            times, classes = zip(*sorted(zip(times, classes)))
            data_class.append(classes)
            
            count = 0
            for cl in classes[:-1]:

                indices=[]
                indices_tst=[]
                
                idx = (digit == cl).nonzero()[0].flatten()
                idx.tolist()
                idxs = set(idx)
                idxs.intersection_update(total_set)
                idx = list(idxs)
                #print(cl,len(idx))

                curr_N = int(len(idx)/3)

                #print(curr_N,N)
                 
                indices.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
                total_set.difference(indices)
                idxs.difference(indices)
                idx = list(idxs)
                indices_tst.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
                total_set.difference(indices_tst)
                
                if curr_N < N:
                    count += (N - curr_N)

                val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
                val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

                tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
                tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

            indices=[]
            indices_tst=[]
            
            idx = (digit == classes[-1]).nonzero()[0].flatten()
            idx.tolist()
            idxs = set(idx)
            idxs.intersection_update(total_set)
            idx = list(idxs)

            indices.extend(list(np.random.choice(idx, size=N+count, replace=False)))
            total_set.difference(indices)
            idxs.difference(indices)
            idx = list(idxs)
            indices_tst.extend(list(np.random.choice(idx, size=N+count, replace=False)))
            total_set.difference(indices_tst)

            val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
            val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

            tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
            tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

        final_lables = [j for sub in data_class for j in sub]
        left = list(total_set)      

    elif data_name == 'census':

        total_set = set(list(np.arange(len(data))))
        
        classes,times = np.unique(data[:,8],return_counts=True) 
        times, classes = zip(*sorted(zip(times, classes)))

        #print(times)
        #print(classes)

        N = int(0.1*len(data)/len(classes))
        
        count = 0
        for cl in classes[:-1]:

            indices=[]
            indices_tst=[]
            
            idx = (data[:,8] == cl).nonzero()[0].flatten()
            idx.tolist()

            curr_N = int(len(idx)/3)

            #print(curr_N,N)
                
            indices.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
            total_set.difference(indices)
            idxs = set(idx)
            idxs.difference(indices)
            idx = list(idxs)
            indices_tst.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
            total_set.difference(indices_tst)
            
            if curr_N < N:
                count += (N - curr_N)

            val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
            val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

            tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
            tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

        indices=[]
        indices_tst=[]
        
        idx = (data[:,8] == classes[-1]).nonzero()[0].flatten()
        idx.tolist()

        indices.extend(list(np.random.choice(idx, size=N+count, replace=False)))
        total_set.difference(indices)
        idxs = set(idx)
        idxs.difference(indices)
        idx = list(idxs)
        indices_tst.extend(list(np.random.choice(idx, size=N+count, replace=False)))
        total_set.difference(indices_tst)

        val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
        val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

        tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
        tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

        #final_lables = [j for sub in data_class for j in sub]
        left = list(total_set)
        
    return data[left], labels[left], val_data_slices, val_label_slices, classes, tst_data_slices,\
        tst_label_slices,classes
