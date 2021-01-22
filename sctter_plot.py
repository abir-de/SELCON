import numpy as np
import os
import sys

import matplotlib
from matplotlib import pyplot as plt
import brewer2mpl
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

'''color_list = [(1, 0, 0),
             (0, 0, 1),
             (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
             (0,0,0),
#              (0.4, 0.7607843137254902, 0.6470588235294118),
             (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
            (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),]'''

def latexify():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,amsfonts}"]
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
    plt.rc('axes', linewidth=1)
    plt.rc('font', weight='bold')
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
    matplotlib.rcParams['lines.markersize']=10 

    
latexify()
fig, ax = plt.subplots()

'''label_map=[r'\textbf{Full}',r'\textbf{Random 0.1\%}', r'\textbf{Random 0.5\%}',r'\textbf{Random 1\%}',\
    r'\textbf{Ours 0.1\%}',r'\textbf{Ours 0.5\%}',r'\textbf{Ours 1\%}'] 

msd_acc = [0.7768764495849609,0.80059802532196,0.769216001033783,0.76904296875,0.76886522769928,\
    0.7690343260765076,0.7691653370857239 ]
msd_time = [105681.917949474,192.184226694,692.4678702260001,1094.4823414359998,2075.55380589,\
    2422.974104494,4355.622487313]
    
for lab in range(len(label_map)):
    plt.scatter(msd_time[lab]/3600, msd_acc[lab], label=label_map[lab], color=color_list[lab])'''

label_map=[r'\textbf{Full}',r'\textbf{Random 0.2\%}', r'\textbf{Random 0.5\%}',r'\textbf{Random 1\%}',\
    r'\textbf{Ours 0.2\%}',r'\textbf{Ours 0.5\%}',r'\textbf{Ours 1\%}'] 

'''time_series_c_acc = [3.544010877609253,22.00385093688965,15.900715827941895,11.865886688232422,\
    14.848151206970215,8.750622749328613,6.11659574508667]
time_series_c_time = [228237.46503130603,1540.703730267,3408.8996684,5805.497203227,7103.005214976001,\
    11690.279037977,16286.553518220002]
    
for lab in range(len(label_map)):
    plt.scatter(time_series_c_time[lab]/3600, time_series_c_acc[lab], label=label_map[lab],\
         color=color_list[lab])'''

time_series_h_acc = [2.7947909832000732,]
time_series_h_time = [227694.306406874,5716.330027417001,]
    
for lab in range(len(label_map)):
    plt.scatter(time_series_h_time[lab]/3600, time_series_h_acc[lab], label=label_map[lab],\
         color=color_list[lab])


plt.ylabel(r'\textbf{Test Error }$\rightarrow$', fontsize=20)

plt.xlabel(r'\textbf{\#Time Taken (in hrs.)}$\rightarrow$', fontsize=20,labelpad=15)
#plt.ylim(top=1,bottom=0)
#plt.xticks(x_axis)#[sel_frac])
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
ax.legend(prop={'size': 18}, frameon=False,handlelength=0.4,loc='center left',\
        bbox_to_anchor=(-0.68, 0.5)) #,'weight':'bold'

plt.box(on=True)
plt.grid(axis='y',linestyle='-', linewidth=1)
plt.grid(axis='x',linestyle='-', linewidth=1)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

file = "results/Images/time_h.png"
plt.savefig(file, bbox_inches='tight')

'''fp = "citeseer_golbal_perm_data"  
data = read_from_pickle(fp)
map_list = data['map_list']
ktau_list = data['ktau_list']
embeds_sens_list = data['embeds_sens_list']

label_map=[r'\textbf{1Perm}',r'\textbf{MultiPerm}', r'\textsc{PermGNN}']    

plt.scatter(ktau_list["LP_LSTMonly_enforceSorted"], map_list["LP_LSTMonly_enforceSorted"], label=label_map[0], c="r")
plt.scatter(ktau_list["LP_LSTMonly_multipermTrain"], map_list["LP_LSTMonly_multipermTrain"], label=label_map[1], c="b")
plt.scatter(ktau_list["LP"], map_list["LP"], label=label_map[2],c="g")'''
