import subprocess
import sys

datadir = '../Datasets/data//NY_Stock_exchange'

datasets = ["NY_Stock_exchange_close"]

fracs =[0.001,0.05,0.01]
num_epochs = 2000#2500#1000
select_every = [35]#,35,50]
reg_lambda = [1e-5]
deltas = [i/10 for i in range(10,0,-1)] #10
past_length = 300

for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for r in reg_lambda:
                for delt in deltas:
                    args = ['python3']
                    args.append('Subset_Whole_Val.py')
                    args.append(datadir)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(str(r))
                    args.append(str(delt))
                    args.append('1')
                    args.append(str(past_length))
                    print(args)
                    subprocess.run(args)
