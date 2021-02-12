import subprocess
import sys

typeOf = str(sys.argv[1])
if_time = sys.argv[2].lower() == 'true'
datadir = sys.argv[3]

datasets = ["NY_Stock_exchange_close"]#'Community_Crime','LawSchool',"cadata","NY_Stock_exchange_close","NY_Stock_exchange_high"]

fracs =[10]#,0.2,0.3,0.4,0.5]
num_epochs = 500
select_every = [35]
reg_lambda = [1e-5]
deltas = [i/10 for i in range(10,0,-1)]
past_length = 100
psuedo_length = 1.0

for dset in datasets:
    for sel in select_every:
        for f in fracs:
            for r in reg_lambda:
                for delt in deltas:
                    args = ['python3']
                    if typeOf == "Fair":
                        args.append('Subset_Fair.py')
                    elif typeOf == "Deep":
                        args.append('Subset_Deep.py')
                    else:
                        args.append('Subset_Main.py')
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
