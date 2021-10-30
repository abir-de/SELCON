import subprocess
import sys

datadir = sys.argv[1]

datasets = ['Community_Crime','LawSchool']

fracs =[0.1]
num_epochs = 2500
select_every = [20,35]
reg_lambda = [1e-5]
deltas = [i/100 for i in range(8,0,-2)]
past_length = 100
psuedo_length = 1.0

for dset in range(len(datasets)):
    for f in fracs:
        for r in reg_lambda:
            for delt in deltas:
                args = ['python3']
                args.append('Subset_Fair.py')
                args.append(datadir + datasets[dset])
                args.append(datasets[dset])
                args.append(str(f))
                args.append(str(num_epochs))
                args.append(str(select_every[dset]))
                args.append(str(r))
                args.append(str(delt))
                args.append('0')
                args.append(str(psuedo_length))
                print(args)
                subprocess.run(args)
