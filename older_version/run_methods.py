import subprocess
import sys

typeOf = str(sys.argv[1])
if_time = sys.argv[2].lower() == 'true'
dset = sys.argv[3] #['LawSchool',"cadata","NY_Stock_exchange_close","NY_Stock_exchange_high"]
datadir = sys.argv[4]

num_epochs = 2000
select_every = [35]
reg_lambda = [1e-5]
past_length = 100
psuedo_length = 1.0

if if_time:
    deltas = [1.0]
    if typeOf != "Deep":
        num_epochs = 5000

    fracs =[0.001,0.003,0.005,0.007,0.01]

else:
    if dset =='LawSchool':
        deltas = [0.04]  #Change this to `deltas = [i/100 for i in range(10,0,-2)]` for delta experiments
    else:
        deltas = [0.3]  #Change this to `deltas = [i/10 for i in range(10,0,-2)]` for delta experiments

    fracs =[0.01,0.03,0.05,0.07,0.1]
                    
for sel in select_every:
    for f in fracs:
        for r in reg_lambda:
            for delt in deltas:
                args = ['python3']
                if typeOf == "Deep":
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
