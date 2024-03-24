#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:47:06 2022

@author: guane
"""

#%% Libraries
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

#%% Load data and variables
df = pd.read_excel("data/Metrics-2.xlsx", index_col=0, sheet_name='Table')
data = np.asarray(df.iloc[1:,2:])

data_base = {'Argone IL':0, 'Beijing':7, 'Chengdu':14 }

names = ['MSE-RNN','MSE-GRU','MSE-LSTM', 'Kernel MSE-RNN', 'Kernel MSE-GRU', 
         'Kernel MSE-LSTM','L1-RNN', 'L1-GRU', 'L1-LSTM', 'Proposal-RNN', 
         'Proposal-GRU', 'Proposal-LSTM']	

metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
values = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

colors = ["black", "dimgray", "grey",
          "blue", "darkblue", "skyblue",
          "green", "lightgreen", "yellowgreen",
          "magenta", "pink", "violet"]


ranking_aux = pd.read_excel("data/Metrics-2.xlsx", index_col=0, sheet_name='Ranking-Individual')
data = np.asarray(df.iloc[1:,2:])

#data_ranking = {'Argone IL':np.asarray(ranking_aux['Ranking Argone'][1:]),
#                'Beijing':np.asarray(ranking_aux['Ranking Beijing'][1:]),
#                'Chengdu':np.asarray(ranking_aux['Ranking Chengdu'][1:]) }

Argone = np.asarray(ranking_aux['Ranking Argone'][1:])
Beijing = np.asarray(ranking_aux['Ranking Beijing'][1:])
Chengdu = np.asarray(ranking_aux['Ranking Chengdu'][1:])


#%% Code
n_rows = len(metrics)
n_cols = len(data_base)
n_methods = 12
size_letters = 30
size_line = 6


j = 0
plt.figure(figsize=(n_cols*12, n_rows*12))

count_ranking = 0
for row in range(n_rows):
    print(' =============================== \n Metrics {} \n =============================== '.format(metrics[row]))
    index_metric = [values[k] + j for k in range(len(values))]    
    #print(index_metric, '\n')    

    for col in range(n_cols):
        
        name = list(data_base.keys())[col] 

        print('\n --------- Data base: {} --------- \n'.format(list(data_base.keys())[col]))
        
        index = n_cols * row + col
        index_init  = data_base[list(data_base.keys())[col]] 
        index_final = index_init + 7        

        
        plt.subplot(n_rows, n_cols, index + 1)
        list_legend = []    
        
        for i in range(len(index_metric)):
            index_model = index_metric[i]
            
            if name == 'Argone IL':
                #name_string = '{modelo}: {ranking}'.format(modelo=names[i], ranking=Argone[index_model] )    
                name_string = '{ranking}'.format(ranking=round(Argone[index_model], 2) )        
            elif name == 'Beijing':
                #name_string = '{modelo}: {ranking}'.format(modelo=names[i], ranking=Beijing[index_model] )    
                name_string = '{ranking}'.format(ranking=round(Beijing[index_model], 2) )    
            elif name == 'Chengdu':
                #name_string = '{modelo}: {ranking}'.format(modelo=names[i], ranking=Chengdu[index_model] )   
                name_string = '{ranking}'.format(ranking=round(Chengdu[index_model], 2) )   
            
            print(name_string)

            plt.plot(data[index_metric[i], index_init:index_final], linestyle='--', linewidth=size_line, label=name_string, color=colors[i])
            plt.yscale('log')
            plt.grid(True)
            plt.ylabel(metrics[row], fontsize=size_letters)
            plt.xlabel('Steps', fontsize=size_letters)
            
            #plt.title(list(data_base.keys())[col], fontsize=20)
            #plt.xlabel('Steps', fontsize=20)
            plt.xticks(np.arange(0,7,1), list(np.arange(1,8,1)), fontsize=size_letters)
            plt.yticks(fontsize=size_letters)
            #plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, fontsize=30) 
            #plt.legend(loc='lower left', borderaxespad=0, fontsize=30) 
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=4, fontsize=size_letters)            
            plt.xlim(0,6)
            
            #list_legend.append(data[index_metric[i], index_init:index_final])
            
            #count +=1
            
        plt.title(label=name, fontsize=size_letters)
        #plt.legend()
    count_ranking += len(names)
        
        
    j += 1

plt.tight_layout(pad=2.0)
plt.show() 

plt.savefig('Resultados.eps', format='eps')




