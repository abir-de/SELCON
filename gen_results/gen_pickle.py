import numpy as np
import os
import sys

import matplotlib
from matplotlib import pyplot as plt
import brewer2mpl

import xlrd

import matplotlib.lines as mlines

import pylab

import pickle


files = [sys.argv[1]]
    
pi_data_name = [sys.argv[2]]

result_dir = "results/Pickle/Noise/"

main_keys =['mean_error','std_dev','time','x-axis']

pi_results = {}

good_labels = ["Random with Constraints","SELCON","Random",\
    "GLISTER","CRAIG","SELCON without Constraints","Full with Constraints","Full"]

file_no = 0

for fil in files:

    #data_name = file.split('/')[-1].split('.')[0]
    
    data_de ={key: {} for key in main_keys}
    acc =[]
    wb = open(fil)
    
    whole_sheet = []

    line = wb.readline()
    while line:
        #print(line)
        whole_sheet.append(line.split('|'))
        line = wb.readline()

    #print(len(whole_sheet))
    for t in [i for i in range(5,10)]+[i for i in range(21,26)]+[i for i in range(-5,0)]:
        #print(whole_sheet[t])
        #if t in [i for i in range(11,14)]:
        #    continue
        whole_sheet[t] = [float(i) for i in whole_sheet[t][:-1]]
    
    #print(whole_sheet[-23:-13])
    #acc = np.array(whole_sheet[-23:-13],dtype=np.float32)

    #print(whole_sheet[4:9])
    time = np.array(whole_sheet[5:10],dtype=np.float32)
    #print(whole_sheet[20:25])
    acc = np.array(whole_sheet[21:26],dtype=np.float32)
    std = np.array(whole_sheet[-5:],dtype=np.float32)

    time[:,0] = time[:,0]*100
    acc[:,0] = acc[:,0]*100
    std[:,0] = std[:,0]*100

    data_de[main_keys[-1]] = time[:,0]

    for i in range(len(good_labels)):
        data_de[main_keys[2]][good_labels[i]] = time[:,i+1] 
        data_de[main_keys[0]][good_labels[i]] = acc[:,i+1]
        #1-acc[:,i+1]/(y_var[pi_data_name[file_no]])
        #/(y_max[pi_data_name[file_no]]**2)
        data_de[main_keys[1]][good_labels[i]] = std[:,i+1]
        #/(y_max[pi_data_name[file_no]]**2) 

    #pi_results[pi_data_name[file_no]] = data_de

    with open(result_dir+pi_data_name[file_no]+'.pkl', 'wb') as output:  
        pickle.dump(data_de, output, pickle.HIGHEST_PROTOCOL)

    file_no+=1