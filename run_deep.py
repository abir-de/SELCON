import subprocess
import sys

if_time = bool(sys.argv[1])

#datadir = '../Datasets/data/NY_Stock_exchange'
#datasets = ["NY_Stock_exchange_close"]
datadir = '../../Datasets/data/'

#datasets = ["MSD"]
datasets = ["LawSchool"]
#datasets = ["cadata"]
#datasets = ["house_pricing"]

#fracs =[0.001,0.003,0.005,0.007,0.01]
#fracs =[0.01,0.03,0.05,0.07,0.1]
fracs =[10]#[0.1,0.2,0.3,0.4,0.5]
num_epochs = 2000#2500#1000
select_every = [35]#,35,50]
reg_lambda = [1e-5]
deltas = [0.3]#[i/10 for i in range(10,0,-1)] #10
past_length = 100

psuedo_length = 1.0

for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for r in reg_lambda:
                for delt in deltas:
                    args = ['python3']
                    args.append('Subset_Deep.py')
                    if if_time:
                        args.append(datadir)
                    else:
                        args.append(datadir + dset)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(str(r))
                    args.append(str(delt))
                    if if_time:
                        args.append('1')
                    else:
                        args.append('0')
                    args.append(str(psuedo_length))
                    if if_time:
                        args.append(str(past_length))
                    print(args)
                    subprocess.run(args)
