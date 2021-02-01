import numpy as np
import os
import sys

import matplotlib
from matplotlib import pyplot as plt
import brewer2mpl

import xlrd

import matplotlib.lines as mlines

import pylab


files = ['results/Lambda/LawSchool/0.05/combined_LawSchool_0.05_all_lambda.txt']#,\
#files = ['results/Lambda/cadata/0.05/combined_cadata_0.05_all_lambda.txt']
#directories = ['results/Delta/LawSchool']
    
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
color_list = bmap.mpl_colors
color_list = [(0.4, 0.7607843137254902, 0.6470588235294118),
             (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
             (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
             (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
             (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
             (1.0, 0.8509803921568627, 0.1843137254901961),
             (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
             (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]

color_list = [(0, 0, 1), 'darkturquoise',(1, 0, 0),'forestgreen','mediumvioletred','darkorange',
            (0,0,0), 'gold','purple',
            (1.0, 0.8509803921568627, 0.1843137254901961),
            (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
            (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
            (0.4, 0.7607843137254902, 0.6470588235294118),
            (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
            (0.7019607843137254, 0.7019607843137254, 0.7019607843137254),
            (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),]

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

color_list = [(0, 0, 1), (1, 0, 0),'darkorange','forestgreen','mediumvioletred',
            (0,0,0),'darkturquoise', 'gold','purple',]

latexify()



sel = 0
x_axis =[]

#good_labels = ["1e-6",'1e-5','1e-4',"1e-3","7e-3","5e-3","3e-3",'1e-2']#,\"Facility with Constraints"],'5e-2'
good_labels = ["1e-6",'1e-5','1e-4',"1e-3",'1e-2']
good_labels = [r'\textbf{'+i+'}' for i in good_labels] 


for fl in files:

    data_name = fl.split('/')[-1]

    wb = open(fl)

    whole_sheet = []

    line = wb.readline()
    while line:
        #print(line)
        whole_sheet.append(line.split('|'))
        line = wb.readline()

    #print(len(whole_sheet))
    for t in range(10,0,-1):
        #print(whole_sheet[-t])
        whole_sheet[-t] = [float(i) for i in whole_sheet[-t][:-1]]
    
    final_acc = np.array(whole_sheet[-10:],dtype=np.float32)

    fig, ax = plt.subplots()
    #ax = fig.add_subplot(2,len(files),len(files)+file_no)
    clr =0

    select = [1,3,5,7,9]

    for i in range(1,final_acc.shape[1]):#):

        ax.plot(final_acc[select,0],final_acc[select,i],label=good_labels[i-1],linewidth=3,markersize=5,\
            marker='o',color=color_list[clr])
        clr+=1

    plt.ylabel(r'\textbf{Test Error }$\rightarrow$', fontsize=20)
    #print(file)

    plt.xlabel(r'\textbf{Delta ($\delta$) }$\rightarrow$', fontsize=20,labelpad=15)
    #plt.ylim(top=1,bottom=0)[1,file_no]
    plt.xticks(final_acc[select,0])
    #print(np.arange(80,97,step=4))
    #plt.yticks(np.arange(80,97,step=4))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    #ax.legend(prop={'size': 18}, frameon=False,handlelength=0.4,loc='center left',\
    #     bbox_to_anchor=(-0.65, 0.5))

    plt.box(on=True)
    plt.grid(axis='y',linestyle='-', linewidth=1)
    plt.grid(axis='x',linestyle='-', linewidth=1)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    file = "results/Images/Lambda/"+data_name+".pdf"#".pdf"
    plt.savefig(file, bbox_inches='tight')
    #file_no+=1
    
fig = pylab.figure()
figlegend = pylab.figure(figsize=(20,1))

my_handles, labels = ax.get_legend_handles_labels()
#figlegend.legend(handles, labels, loc='upper center')

figlegend.legend(ncol=5, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
                 frameon=False,handles=my_handles,
                               handlelength=0.9,fontsize=100,handletextpad=0.3,columnspacing=2.0)

file = "results/Images/Lambda/legend.png"#".pdf"
plt.savefig(file, bbox_inches='tight')
