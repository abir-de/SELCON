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

bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)

def latexify():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,amsfonts}"]
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
    plt.rc('axes', linewidth=1)
    plt.rc('font', weight='bold')
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']


latexify()


color_list = [(0, 0, 1),(1, 0, 0), (0,0,0),'purple','forestgreen','darkorange']

good_labels = ["Random with Constraints",'SELCON','Full with Contraints','Full','Random',\
    'SELCON without Constraints']#,\"Facility with Constraints"]

good_labels_pi = ["Random with Constraints",'SELCON','Full with Contraints','Random',\
    'SELCON without Constraints']
good_labels = [r'\textbf{'+i+'}' for i in good_labels] 

pi_data_name = ['Comm_Crime','LawSchool']

main_keys =['mean_error','std_dev','Delta']

pi_results = {}

result_dir = "results/Pickle/"

files = ['results/Slice/Community_Crime/0.1/combined_Community_Crime_0.1.txt',\
    'results/Slice/LawSchool/0.1/combined_LawSchool_0.1.txt']

file_no = 0
for file in files:

    data_name = file.split('/')[-1].split('.')[0]
    
    data_de ={key: {} for key in main_keys}
    
    acc =[]
    wb = open(file)
    whole_sheet = []
    line = wb.readline()
    while line:
        #print(line)
        whole_sheet.append(line.split('|'))
        line = wb.readline()

    for t in range(23,0,-1):
        #print(whole_sheet[-t])
        if t in [i for i in range(11,14)]:
            continue
        whole_sheet[-t] = [float(i) for i in whole_sheet[-t][:-1]]
    
    acc = np.array(whole_sheet[-23:-13],dtype=np.float32)

    std = np.array(whole_sheet[-10:],dtype=np.float32)

    data_de[main_keys[-1]] = acc[:,0]

    print(acc.shape)

    select = [1,2,3,5,acc.shape[1]-1]

    for i in range(len(good_labels_pi)):

        data_de[main_keys[0]][good_labels_pi[i]] = acc[:,select[i]] 
        data_de[main_keys[1]][good_labels_pi[i]] = std[:,select[i]] 

    with open(result_dir+pi_data_name[file_no]+'_fair.pkl', 'wb') as output:  
        pickle.dump(data_de, output, pickle.HIGHEST_PROTOCOL)
    
    file_no+=1

    fig, ax = plt.subplots()
    #ax = fig.add_subplot(2,len(files),len(files)+file_no)
    clr =0

    select = [1,2,3,4,5,acc.shape[1]-1]

    for i in select:#):
       
        ax.plot(acc[1:,0],acc[1:,i],label=good_labels[clr],linewidth=3,markersize=5,\
            marker='o',color=color_list[clr])
        clr+=1

    #plt.ylabel(r'\textbf{Time (in '+time_unit_str+r')}$\rightarrow$', fontsize=20)
    plt.ylabel(r'\textbf{Mean Error}$\rightarrow$', fontsize=20)
    print(file)

    plt.xlabel(r'\textbf{Delta ($\delta$) }$\rightarrow$', fontsize=20,labelpad=15)
    #plt.ylim(top=1,bottom=0)[1,file_no]
    plt.xticks(acc[:,0])
    #print(np.arange(80,97,step=4))
    #plt.yticks(np.arange(80,97,step=4))
    plt.yscale("log")
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    #ax.legend(prop={'size': 18}, frameon=False,handlelength=0.4,loc='center left',\
    #     bbox_to_anchor=(-0.65, 0.5))

    #ax.legend(ncol=5, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
    #             frameon=False,handles=[DG,SDG,Triage_Alg,Triage_Est,FA,NA],
    #                           handlelength=0.7,fontsize=70,handletextpad=0.3,columnspacing=3.0)

    plt.box(on=True)
    plt.grid(axis='y',linestyle='-', linewidth=1)
    plt.grid(axis='x',linestyle='-', linewidth=1)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    file = "results/Images/Fairness/"+data_name+".pdf"#".pdf"
    plt.savefig(file, bbox_inches='tight')
    #file_no+=1


fig = pylab.figure()
figlegend = pylab.figure(figsize=(20,1))

my_handles, labels = ax.get_legend_handles_labels()
#figlegend.legend(handles, labels, loc='upper center')

figlegend.legend(ncol=5, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
                 frameon=False,handles=my_handles,
                               handlelength=0.9,fontsize=100,handletextpad=0.3,columnspacing=2.0)

file = "results/Images/Fairness/legend.pdf"#".pdf"
plt.savefig(file, bbox_inches='tight')