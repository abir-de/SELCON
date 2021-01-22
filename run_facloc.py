import subprocess
import sys

datadir = '../Datasets/data//NY_Stock_exchange'

datasets = ["NY_Stock_exchange_close"]

fracs =[0.001,0.004,0.05,0.006,0.01]
#num_epochs = 2000#2500#1000
#select_every = [35]#,35,50]
#reg_lambda = [1e-5]
#deltas = [i/10 for i in range(10,0,-1)] #10
past_length = 300

for dset in datasets:
    #for f in fracs:
            
    args = ['python3']
    args.append('facility_location.py')
    args.append(datadir)
    args.append(dset)
    args.append(str(fracs))
    args.append('1')
    args.append(str(past_length))
    print(args)
    subprocess.run(args)
