#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 16:06:55 2023

@author: guane
"""

#%% Libraries 
import pickle 
import os 
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset


#%% load data 
with open ('./pkl/Beijing_Airport_China.pkl', 'rb') as input_file:
    data = pickle.load(input_file)

#%% Plot

for predictionHorizon in [0,3,6]:
    print(predictionHorizon+1)   

    fig = plt.figure(figsize=(15, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.plot(data['Test'][:,predictionHorizon], c='k', label='Target', linestyle='--')
    ax0.plot(data['RNN'][:,predictionHorizon], c='b', label='RNN', linestyle='--')
    ax0.plot(data['GRU'][:,predictionHorizon], c='r', label='GRU', linestyle='--')
    ax0.plot(data['LSTM'][:,predictionHorizon], c='g', label='LSTM', linestyle='--')    
    plt.xlim(0,data['Test'][:,predictionHorizon].shape[0])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Wind speed', fontsize=15)

    ax1 = plt.subplot(gs[1])
    ax1.plot(data['Test'][:,predictionHorizon][0:300], c='k', label='Target', linestyle='--')
    ax1.plot(data['RNN'][:,predictionHorizon][0:300], c='b', label='RNN', linestyle='--')
    ax1.plot(data['GRU'][:,predictionHorizon][0:300], c='r', label='GRU', linestyle='--')
    ax1.plot(data['LSTM'][:,predictionHorizon][0:300], c='g', label='LSTM', linestyle='--')    
    ax1.axes.yaxis.set_ticklabels([])
    plt.xlim(0,300)
    plt.xticks(fontsize=15)
    
    plt.tight_layout()    
    
    plt.tight_layout(pad=3.0)    
    plt.show()
    #plt.savefig('./results/{name}.png'.format(name=str(predictionHorizon+1)), format='png')


