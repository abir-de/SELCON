import numpy as np
import matplotlib.pyplot as plt 
import os
import sys

directory = sys.argv[1]
data_name = sys.argv[2]

#data_name1 = data_name+"_delta" #sys.argv[2].split('.')[0]
#data_name2 = data_name+"_lamda" #sys.argv[3].split('.')[0]

#file = os.path.join(directory,data_name1+".txt")

#file2 = os.path.join(directory,data_name2+".txt")

colors = ['g-','c-','b-','r-','orange','#000000','#8c564b','m-','y','pink']
labels =['lamda','Test','Validation','Objective','Training error','Regularizer']

'''data_name1 = data_name+"_delta="
files = [data_name1+str(i/100)+'.txt' for i in range(102,96,-1)]
files.insert(0,data_name1+str(1.05)+'.txt')'''

files = [f for f in os.listdir(directory) if f.endswith(".txt")]
#print(files)

table = [[] for i in range(len(files))]

ta = 0
max_len = 0
max_id = 0
x_axis = []
for file in files:
    #print(file.split('=')[1][:-4])
    val = file.split('=')[1][:-4]
    x_axis.append(float(val))
    data = []
    with open(os.path.join(directory,file)) as fp:

        line = fp.readline()
        while line:

            '''info = [float(i.strip()) for i in line.strip().split(",")]

            for i in range(len(info)):

                data[i].append(info[i])'''

            data.append([float(i.strip()) for i in line.strip().split(",")])

            line = fp.readline()
    
    table[ta] = data
    if max_len < len(data):
        max_len = len(data)
        max_id = ta
    ta +=1


#labels =['epsilon_i','Test','Validation','Objective','Training error','Regularizer']

plt.figure()

#table = np.array(table)
print(len(table))

deltas = [[] for i in range(max_len)]
x_axis_list = [[] for i in range(max_len)]


for de in range(max_len):
    #print(table[max_id][de][0])
    for j in range(len(table)):
        for k in range(len(table[j])):

            if table[max_id][de][0] == table[j][k][0]:
                deltas[de].append(table[j][k])
                x_axis_list[de].append(x_axis[j])
                break
                #if table[max_id][de][0] == 0.1:
                #    print(table[j][k])

    deltas[de] = np.array(deltas[de]) 

#for de in range(len(deltas)):
#    deltas[de] = np.array(deltas[de])
#print(deltas[24])

'''x_axis = []
for f in files:
    val = f.split('=')[1][:-4]
    #print(val)
    x_axis.append(float(val))'''

#x_axis = sorted(x_axis)
    
#x_axis = [i/100 for i in range(102,96,-1)]
#x_axis.insert(0,1.05)

for de in range(len(deltas)):
    
    cl =0
    for i in [1,2,4]:#range(len(labels)):
        list1, list2 = zip(*sorted(zip(x_axis_list[de], deltas[de][:,i]))) #,reverse=True
        #print(deltas[de][0,0],list2)
        plt.plot(list1,list2,colors[cl],label = labels[i]) #colors[cl],'o',
        cl+=1

    plt.legend() 
    plt.xlabel('Delta') 
    plt.ylabel('Error')
    #plt.xscale('log')
    plt.title("Lamda = "+str(deltas[de][0,0])+" vs Error") 
    plt.savefig(directory+"/lambda/"+data_name+'_'+str(deltas[de][0,0])+'.png')
    plt.clf()


#data_name2 = data_name+"_lamda" #sys.argv[3].split('.')[0]

file2 = os.path.join(directory,data_name+"_delta=1.txt")
#files = [data_name1+str(i/100)+'.txt' for i in range(102,96,-1)]
#files.insert(0,data_name1+str(1.05)+'.txt')

table = [[] for i in range(6)]
with open(file2) as fp:

    line = fp.readline()
    while line:

        info = [float(i.strip()) for i in line.strip().split(",")]

        for i in range(len(info)):

            table[i].append(info[i])

        line = fp.readline()

table = np.array(table)
#print(table.shape)

plt.figure()
data_name2 = "For delta=1, Lambda"#"For sigma=30 lambda"
cl =0
for i in [1,2,4]:#range(len(labels)):
    plt.plot(table[0,:], table[i,:],colors[cl],label = labels[i])
    cl+=1

plt.legend() 
plt.xlabel('lambda') 
plt.ylabel('Error')
plt.xscale('log')
plt.title(data_name2+" vs Error") 
plt.savefig(directory+"/"+data_name2+'.png')
plt.clf()

plt.figure()

