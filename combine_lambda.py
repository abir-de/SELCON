import numpy as np
import os
import sys
import copy

data_name = sys.argv[2]
directory = sys.argv[1]
frac= sys.argv[3]

in_dir = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

fractions = [ int(10e6*round(float(i), 6)) for i in in_dir]
fractions, in_dir = zip(*sorted(zip(fractions, in_dir)))

logfile = open(os.path.join(directory, 'combined_'+data_name+'_'+frac+'_all_lambda.txt'), 'w')

selection =[]
for di in in_dir:
	temp = [float(i) for i in os.listdir(os.path.join(directory, di))]
	if len(selection) < len(temp): 
		selection = copy.deepcopy(temp)

deltas = [ int(10e2*round(float(i), 2)) for i in selection]
deltas, selection = zip(*sorted(zip(deltas, selection)))

lam_val_acc = []
lam_test_acc = []
lam_time =[]

for sel in selection:

    #print("\nSelect every",sel,file=logfile)

    val_acc = [[] for _ in range(len(in_dir)+1)]
    test_acc = [[] for _ in range(len(in_dir)+1)]
    time =[[] for _ in range(len(in_dir)+1)]

    first = True

    time[0].append(" ")
    val_acc[0].append(" ")
    test_acc[0].append(" ")

    for delta in range(len(in_dir)):

        file_path = os.path.join(directory,in_dir[delta],str(sel),'35',data_name+'.txt')

        print(file_path)

        if not os.path.exists(file_path):
            continue

        print(file_path)

        with open(file_path) as fp:
            
            time[delta+1].append(float(in_dir[delta]))
            val_acc[delta+1].append(float(in_dir[delta]))
            test_acc[delta+1].append(float(in_dir[delta]))

            line = fp.readline()

            #print(line)

            while line:

                tim = [i.strip() for i in line.strip().split(" ")]

                if tim[0] in ["Subset","Random","Full","Facility","Glister"]:

                    #print(line)

                    if len(tim) > 3:
                        if first:
                            if tim[2] in ["Constraints","fairness"]:
                                time[0].append(tim[0]+" "+tim[1]+" "+tim[2])
                            elif tim[3] in ["Constraints","fairness"]:
                                time[0].append(tim[0]+" "+tim[2]+" "+tim[3])
                            else:
                                time[0].append(tim[0])
                        time[delta+1].append(float(tim[-1]))

                    else:
                        if first:
                            if len(tim) == 3:
                                val_acc[0].append(tim[0]+" "+tim[1]+" "+tim[2])
                                test_acc[0].append(tim[0]+" "+tim[1]+" "+tim[2])
                            else:
                                val_acc[0].append(tim[0])
                                test_acc[0].append(tim[0])
                        
                        line = fp.readline()
                        acc = [i.strip() for i in line.strip()[:-1].split("|")]
                        val_acc[delta+1].append(float(acc[-1]))

                        line = fp.readline()
                        acc = [i.strip() for i in line.strip()[:-1].split("|")]
                        test_acc[delta+1].append(float(acc[-1])) 

                line = fp.readline()
        
        first = False

    lam_test_acc.append(test_acc)
    lam_val_acc.append(val_acc)
    lam_time.append(time)

lambda_val = []

for i in range(len(lam_time[0])):
    lambda_val.append(lam_time[0][i][0])

top_line = ""
for i in lambda_val:
    top_line = top_line+str(i)+"|"


if len(val_acc) > 0:

    for base in range(1,len(lam_time[0][0])):

        print("\n"+lam_time[0][0][base],file=logfile)
        print("\nTime",file=logfile)

        print("\n"+top_line,file=logfile)
        
        for lt in range(len(lam_time)):
            line =  str(selection[lt])+"|"
            for la in range(1,len(lam_time[lt])):
                line = line + str(lam_time[lt][la][base])+"|"    
            print(line,file=logfile)
        
        print("\nValidation Accuracies",file=logfile)
        
        print("\n"+top_line,file=logfile)
        
        for lt in range(len(lam_val_acc)):
            line = str(selection[lt])+"|"
            for la in range(1,len(lam_val_acc[lt])):
                line = line + str(lam_val_acc[lt][la][base])+"|"    
            print(line,file=logfile)

        print("\nTest Accuracies",file=logfile)

        print("\n"+top_line,file=logfile)
        
        for lt in range(len(lam_test_acc)):
            line =  str(selection[lt])+"|"
            for la in range(1,len(lam_test_acc[lt])):
                line = line + str(lam_test_acc[lt][la][base])+"|"    
            print(line,file=logfile)
