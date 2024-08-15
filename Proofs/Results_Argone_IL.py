

# Libraries
import os, pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Weights 
with open('./Results-Argone_IL.pkl', 'rb') as f:
    data = pickle.load(f)   # RNNSimple, GRU, LSTM

"""
    data[A][B][C]
        - A: model
            RNNSimple, GRU, LSTM
        - B: Threshold
            1 , 10, 100, 500, 1000
        - C: Values
            results, weights
            * results (10,5,7)
                10: folds 
                5: metrics (MSE, RMSE, MAE, MAPE, R2) 
                7: Maximum prediction horizon
            * weights  
"""

results = data["RNNSimple"][1]['results']


