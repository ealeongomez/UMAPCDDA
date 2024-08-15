#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 08:48:03 2022

@author: guane
"""

#%% Libraries
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

#%% Load data and variables
df = pd.read_excel("data/Metrics-2.xlsx", index_col=0, sheet_name='Table')
data = np.asarray(df.iloc[1:,2:])

names = ['MSE-RNN','MSE-GRU','MSE-LSTM', 'Kernel MSE-RNN', 'Kernel MSE-GRU', 
         'Kernel MSE-LSTM','L1-RNN', 'L1-GRU', 'L1-LSTM', 'Proposal-RNN', 
         'Proposal-GRU', 'Proposal-LSTM']	

metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
values = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

ranking_aux = pd.read_excel("data/Metrics-2.xlsx", index_col=0, sheet_name='Ranking-Individual')
data_base = {'Argone IL':np.asarray(ranking_aux['Ranking Argone'][1:]),
             'Beijing':np.asarray(ranking_aux['Ranking Beijing'][1:]),
             'Chengdu':np.asarray(ranking_aux['Ranking Chengdu'][1:]) }

#%% Code

n_rows = len(metrics)
n_cols = len(data_base)
n_methods = 12

j = 0
plt.figure(figsize=(n_cols*10, n_rows*6))

for row in range(n_rows):
    print(' \n ------------------- Metrics {} ------------------- '.format(metrics[row]))
    index_metric = [values[k] + j for k in range(len(values))]    
    print(index_metric, '\n')    
    
    for col in range(n_cols):
        print('\n Data base: {} \n'.format(list(data_base.keys())[col]))
        
        index = n_cols * row + col
        index_init  = data_base[list(data_base.keys())[col]] 
        index_final = index_init + 7
        
        #print(index_init, index_final)
        
        plt.subplot(n_rows, n_cols, index + 1)
        list_legend = []    
        for i in range(len(index_metric)):
            index_model = index_metric[i]

            #name_string = '{modelo}: {ranking}'.format(modelo=names[i], ranking=round(ranking[index_metric[i]],2) )
            #name_string = str(round(ranking[index_metric[i]],2))
            name_string = ' '
            
            #print(name_string, data[index_metric[i], index_init:index_final])
            print(index_metric[i])

            plt.plot(data[index_metric[i], index_init:index_final], linestyle='--', linewidth=6.0, label=name_string)
            #plt.plot(data[index_metric[i], index_init:index_final], linestyle='--')
            plt.yscale('log')
            plt.grid(True)
            plt.ylabel(metrics[row], fontsize=30)
            #plt.title(list(data_base.keys())[col], fontsize=20)
            #plt.xlabel('Steps', fontsize=20)
            plt.xticks(np.arange(0,7,1), list(np.arange(1,8,1)), fontsize=30)
            plt.yticks(fontsize=30)
            #plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, fontsize=7) 
            plt.xlim(0,6)
            
            #list_legend.append(data[index_metric[i], index_init:index_final])
            
        plt.title(label=names[index], fontsize=30)
        #plt.legend()
        
    j += 1

plt.tight_layout(pad=2.0)
plt.show() 





