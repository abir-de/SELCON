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
color_list = bmap.mpl_colors
color_list = [(0.4, 0.7607843137254902, 0.6470588235294118),
             (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
             (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
             (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
             (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
             (1.0, 0.8509803921568627, 0.1843137254901961),
             (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
             (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]

color_list = [(0, 0, 1),(1, 0, 0),'forestgreen','mediumvioletred','darkorange',
            (0,0,0),'darkturquoise', 'gold','purple',
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


latexify()

sel = 0
x_axis =[]

directories = ['results/Delta/cadata','results/Delta/LawSchool']
#directories = ['results/Delta/LawSchool']

good_labels = ['3\%','7\%','10\%','Full']#,\"Facility with Constraints"] '1\%','3\%',
good_labels_pi  =  ['3\%','7\%','10\%','Full']
good_labels = [r'\textbf{'+i+'}' for i in good_labels] 

pi_data_name = ['cadata','LawSchool']

main_keys =['mean_error','std_dev','Delta']

pi_results = {}

result_dir = "results/Pickle/"

file_no = 0

for ndir in directories:

    data_name = ndir.split('/')[-1]

    in_dir = [i for i in  os.listdir(ndir)]
    fractions = [ int(100*round(float(i), 3)) for i in  os.listdir(ndir)]
    fractions, in_dir = zip(*sorted(zip(fractions, in_dir)))

    final_acc = np.zeros((10,len(fractions)+1))
    final_std = np.zeros((10,len(fractions)+1))
    frac_no=1
    #print(in_dir)
    data_de ={key: {} for key in main_keys}
    for frac in in_dir:
        frac_path = os.path.join(ndir,frac)

        onlyfiles = [f for f in os.listdir(frac_path) if os.path.isfile(os.path.join(frac_path, f)) and f.endswith(".txt")]

        #print(onlyfiles)
        wb = open(os.path.join(frac_path,onlyfiles[0]))

        print(os.path.join(frac_path,onlyfiles[0]))
        
        whole_sheet = []

        line = wb.readline()
        while line:
            #print(line)
            whole_sheet.append(line.split('|'))
            line = wb.readline()

        #print(len(whole_sheet))
        for t in range(23,0,-1):
            #print(whole_sheet[-t])
            if t in [i for i in range(11,14)]:
                continue
            whole_sheet[-t] = [float(i) for i in whole_sheet[-t][:-1]]
        
        #print(whole_sheet[-23:-13])
        acc = np.array(whole_sheet[-23:-13],dtype=np.float32)

        final_acc[:,0] = acc[:,0]
        final_acc[:,frac_no] = acc[:,-1]

        std = np.array(whole_sheet[-10:],dtype=np.float32)

        final_std[:,0] = std[:,0]
        final_std[:,frac_no] = std[:,-1]

        frac_no += 1

        #time = np.array(whole_sheet[4:9],dtype=np.float32)
        #acc = np.array(whole_sheet[-5:],dtype=np.float32)

        #print(time)
        #print(acc)

        #ax = fig.add_subplot(1,len(files),file_no)

    data_de[main_keys[-1]] = final_acc[:,0]

    #print(final_acc.shape)
    
    bud = [2,4,5,6]
    for i in range(len(good_labels)):
        data_de[main_keys[0]][good_labels_pi[i]] = final_acc[:,bud[i]] 
        data_de[main_keys[1]][good_labels_pi[i]] = final_std[:,bud[i]] 

    with open(result_dir+pi_data_name[file_no]+'_delta.pkl', 'wb') as output:  
        pickle.dump(data_de, output, pickle.HIGHEST_PROTOCOL)
    
    print(frac_no,final_acc.shape)

    fig, ax = plt.subplots()
    #ax = fig.add_subplot(2,len(files),len(files)+file_no)
    clr =0

    select = [1,3,5,7,9]

    start = 2    

    for i in range(start,final_acc.shape[1]):#):

        if i==3:
            continue

        ax.plot(final_acc[select,0],final_acc[select,i],label=good_labels[clr],linewidth=3,\
            markersize=5,marker='o',color=color_list[clr])
        clr+=1

    plt.ylabel(r'\textbf{Test Error }$\rightarrow$', fontsize=20)
    #print(file)

    plt.xlabel(r'\textbf{Delta ($\delta$) }$\rightarrow$', fontsize=20,labelpad=15)
    #plt.ylim(top=1,bottom=0)[1,file_no]
    plt.xticks(acc[select,0])
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

    file = "results/Images/Delta_vs_Acc/"+data_name+".pdf"#".pdf"
    plt.savefig(file, bbox_inches='tight')
    file_no+=1
    
fig = pylab.figure()
figlegend = pylab.figure(figsize=(20,1))

my_handles, labels = ax.get_legend_handles_labels()
#figlegend.legend(handles, labels, loc='upper center')

figlegend.legend(ncol=5, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
                 frameon=False,handles=my_handles,
                               handlelength=0.9,fontsize=100,handletextpad=0.3,columnspacing=2.0)

file = "results/Images/Delta_vs_Acc/legend.pdf"#".pdf"
plt.savefig(file, bbox_inches='tight')


'''color_list = [(0, 0, 1), (1, 0, 0),'darkorange','forestgreen','mediumvioletred',
            (0,0,0),'darkturquoise', 'gold','purple',]


good_labels = ["Random with Constraints",'SELCON with Constraints','Full with Contraints']#,\"Facility with Constraints"]
good_labels = [r'\textbf{'+i+'}' for i in good_labels] 


for ndir in directories:

    data_name = ndir.split('/')[-1]

    in_dir = [i for i in  os.listdir(ndir)]
    fractions = [ int(100*round(float(i), 3)) for i in  os.listdir(ndir)]
    fractions, in_dir = zip(*sorted(zip(fractions, in_dir)))

    sub_final_acc = np.zeros((10,len(fractions)))
    rand_final_acc = np.zeros((10,len(fractions)))

    full_acc = np.zeros((10,2))

    frac_path = os.path.join(ndir,in_dir[-1])
    onlyfiles = [f for f in os.listdir(frac_path) if os.path.isfile(os.path.join(frac_path, f)) and f.endswith(".txt")]
    wb = open(os.path.join(frac_path,onlyfiles[0]))
    
    whole_sheet = []
    line = wb.readline()
    while line:
        #print(line)
        whole_sheet.append(line.split('|'))
        line = wb.readline()

    for t in range(10,0,-1):
        #print(whole_sheet[-t])
        whole_sheet[-t] = [float(i) for i in whole_sheet[-t][:-1]]
    
    acc = np.array(whole_sheet[-10:],dtype=np.float32)

    full_acc[:,0] = acc[:,0]
    full_acc[:,1] = acc[:,-1]
    
    
    frac_no=1
    for frac in in_dir[:-1]:
        frac_path = os.path.join(ndir,frac)

        onlyfiles = [f for f in os.listdir(frac_path) if os.path.isfile(os.path.join(frac_path, f)) and f.endswith(".txt")]

        #print(onlyfiles)
        wb = open(os.path.join(frac_path,onlyfiles[0]))

        print(os.path.join(frac_path,onlyfiles[0]))
        
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
        
        acc = np.array(whole_sheet[-10:],dtype=np.float32)

        sub_final_acc[:,0] = acc[:,0]
        sub_final_acc[:,frac_no] = acc[:,-1]

        rand_final_acc[:,0] = acc[:,0]
        rand_final_acc[:,frac_no] = acc[:,-2]

        fig, ax = plt.subplots()
        #ax = fig.add_subplot(2,len(files),len(files)+file_no)
        #clr =0

        ax.plot(rand_final_acc[1:,0],rand_final_acc[1:,frac_no],label=good_labels[0],linewidth=3,\
            markersize=5,marker='o',color=color_list[0])

        ax.plot(sub_final_acc[1:,0],sub_final_acc[1:,frac_no],label=good_labels[1],linewidth=3,\
            markersize=5,marker='o',color=color_list[1])

        ax.plot(full_acc[1:,0],full_acc[1:,1],label=good_labels[2],linewidth=3,\
            markersize=5,marker='o',color=color_list[2])

        plt.ylabel(r'\textbf{Test Error }$\rightarrow$', fontsize=20)
        #print(file)

        plt.xlabel(r'\textbf{Delta ($\delta$) }$\rightarrow$', fontsize=20,labelpad=15)
        #plt.ylim(top=1,bottom=0)[1,file_no]
        plt.xticks(acc[1:,0]) #[0,1,3,5,6,8,9]
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

        file = "results/Images/Delta_vs_Acc/"+data_name+"_"+frac+".pdf"#".pdf"
        plt.savefig(file, bbox_inches='tight')
        frac_no += 1
    
fig = pylab.figure()
figlegend = pylab.figure(figsize=(20,1))

my_handles, labels = ax.get_legend_handles_labels()
#figlegend.legend(handles, labels, loc='upper center')

figlegend.legend(ncol=5, mode="expand", borderaxespad=0.,prop={'size': 25,'weight':200},
                 frameon=False,handles=my_handles,
                               handlelength=0.9,fontsize=100,handletextpad=0.3,columnspacing=2.0)

file = "results/Images/Delta_vs_Acc/legend.pdf"#".pdf"
plt.savefig(file, bbox_inches='tight')'''




