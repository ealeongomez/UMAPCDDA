#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:07:07 2022

@author: E.A. León-Gómez
"""

#%% Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns

from scipy import stats
import scikit_posthocs as sp 
import stac
import matplotlib.pyplot as plt

location = os.getcwd()

#%% Load data and variables

df = pd.read_excel("Data/Metrics.xlsx", index_col=0, sheet_name='Table')
data = np.asarray(df.iloc[1:,2:])

metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
values = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

names = ['MSE-RNN','MSE-GRU','MSE-LSTM', 'Kernel MSE-RNN', 'Kernel  MSE-GRU', 
         'Kernel  MSE-LSTM','L1-RNN', 'L1-GRU', 'L1-LSTM', 'Proposal-RNN', 
         'Proposal-GRU', 'Proposal-LSTM']		

results_ranking = np.zeros(data.shape)

#%% Code

# Prediction horizon models
prediction = 1
for i in range(data.shape[1]):
    print("\n Prediction horizon {horizon} ------------------ ".format(horizon=str(prediction)))        
    if prediction == 7: 
        prediction = 1
    else:
        prediction += 1         
  
    
    # Performance metrics  
    for j in range(len(metrics)):
        print('Performance {}'.format(metrics[j]))
        index = [values[k] + j for k in range(len(values))]
        
        data_ranking = np.array([data[int(index[q])][i] for q in range(len(index))])
        values_for_ranking = pd.Series(data_ranking, index=names)
        
        ser = pd.Series(data_ranking, index=names).rank(ascending=True) 
                
        # Save results ------------------------------------------------------
        for s in range(len(ser)):
            #print(i, index[s], ser[s])            
            results_ranking[index[s]][i] = int(ser[s])
            
                        

        
        
        
    
    