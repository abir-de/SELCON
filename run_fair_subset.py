import subprocess

datadir = '../Datasets/data/'
#datasets = [ 'census']
#datasets = [ 'Community_Crime']
#datasets = ['OnlineNewsPopularity']
#datasets = ["German_credit"]
datasets = ["synthetic"]
#fracs =[0.1,0.2]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#fracs =[ 0.5,0.6,0.7,0.8,0.9]
fracs =[0.1,0.2,0.3,0.4,0.5]
num_epochs = 100#2500#1000
select_every = [20]#,35,50]
reg_lambda = [0.01]
deltas = [i/10 for i in range(10,0,-1)]

for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for r in reg_lambda:
                for delt in deltas:
                    args = ['python3']
                    args.append('Subset_Whole_Val.py')
                    args.append(datadir + dset)
                    args.append(dset)
                    args.append(str(f))
                    args.append(str(num_epochs))
                    args.append(str(sel))
                    args.append(str(r))
                    args.append(str(delt))
                    print(args)
                    subprocess.run(args)
