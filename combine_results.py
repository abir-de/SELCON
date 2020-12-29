import numpy as np
import os
import sys
import copy


data_name = sys.argv[2]
directory = sys.argv[1]

in_dir = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

#fractions = [ int(100*round(float(i), 2)) for i in in_dir]
#fractions, in_dir = zip(*sorted(zip(fractions, in_dir)))

selection =[]
for di in in_dir:
	temp = [int(i) for i in os.listdir(os.path.join(directory, di))]
	if len(selection) < len(temp): 
		selection = copy.deepcopy(temp)

for sel in selection:

    val_acc = [[] for _ in range(len(in_dir)+1)]
    test_acc = [[] for _ in range(len(in_dir)+1)]
    time =[[] for _ in range(len(in_dir)+1)]

    first = True

    time[0].append(" ")

    for delta in range(len(in_dir)):

        file_path = os.path.join(directory,in_dir[delta],str(sel),data_name+'.txt')

        if not os.path.exists(file_path):
            continue

        with open(file_path) as fp:
            line = fp.readline()

            while line:

                tim = [i.strip() for i in line.strip().split(" ")]

                if tim[0] in ["Subset","Random","Full"]:

                    if len(tim) > 3:
                        if first:
                            if tim[2] in ["Constraints","fairness"]:
                                time[0].append(tim[0]+tim[1]+tim[2])
                            else:
                                time[0].append(tim[0])
                        time[delta].append(float(tim[-1]))

                    else:
                        if first:
                            if len(tim) == 3:
                                val_acc[0].append(tim[0]+tim[1]+tim[2])
                                test_acc[0].append(tim[0]+tim[1]+tim[2])
                            else:
                                val_acc[0].append(tim[0])
                                test_acc[0].append(tim[0])
                        
                        line = fp.readline()
                        acc = [i.strip() for i in line.strip()[:-1].split("|")]
                        val_acc[delta].append(float(acc[-1]))

                        line = fp.readline()
                        acc = [i.strip() for i in line.strip()[:-1].split("|")]
                        test_acc[delta].append(float(acc[-1]))                    

                line = fp.readline()
        
        first = False
