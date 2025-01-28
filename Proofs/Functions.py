# ==================================================================
# Libraries
# ==================================================================
# Basic
from tabnanny import verbose
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from tabulate import tabulate

# Deep larning
import tensorflow as tf
from tensorflow import keras

# Métricas de rendimiento
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================================================================
# Class
# ==================================================================
 
class ForecastingModels():                              

    def __init__(self, X, y, X_test, y_test, neurons, batch_size, epochs, n_splits): 
        self.X = X
        self.y = y
        self.X_test = X_test        
        self.y_test = y_test
        self.neurons = neurons
        self.batch_size = batch_size
        self.epochs = epochs  
        self.n_splits = n_splits
        try:
            self.predictionHorizon = self.y.shape[1] 
        except:
            self.predictionHorizon = 1

    def mean_absolute_percentage_error(self, forecasting_test):
        mape = np.mean(np.abs((self.y_test - forecasting_test)/self.y_test))        
        return mape

    """ Recurrent Neural Network Simple """
    def RNNSimple_Model(self):      
      
      # Arquitecture 
      input = tf.keras.layers.Input(shape=(self.X.shape[1], self.X.shape[2]), name='Input')
      rnn_1 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h1')(input)
      rnn_2 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h2')(rnn_1)
      rnn_3 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h3')(rnn_2)
      rnn_4 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h4')(rnn_3)
      output = keras.layers.SimpleRNN(self.predictionHorizon)(rnn_4)

      # Create model
      self.model_RNNSimple = tf.keras.Model(inputs=input, outputs=output)

      return self.model_RNNSimple
       
    """ Gate Recurrent Unit """
    def GRU_Model(self):
        
        # Arquitecture
        input = tf.keras.layers.Input(shape=(self.X.shape[1], self.X.shape[2]), name='Input')
        gru_1 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h1')(input)
        gru_2 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h2')(gru_1)
        gru_3 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h3')(gru_2)
        gru_4 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h4')(gru_3)
        output = keras.layers.GRU(self.predictionHorizon)(gru_4)

        # Create model
        self.model_GRU = tf.keras.Model(inputs=input, outputs=output)

        return self.model_GRU
    
    
    def LSTM_Model(self):

        # Arquitecture        
        input_layer = tf.keras.layers.Input(shape=(self.X.shape[1], self.X.shape[2]))
        lstm_1 = tf.keras.layers.LSTM(self.neurons, return_sequences=True)(input_layer)        
        lstm_2 = tf.keras.layers.LSTM(self.neurons, return_sequences=True)(lstm_1)
        lstm_3 = tf.keras.layers.LSTM(self.neurons, return_sequences=True)(lstm_2)
        lstm_4 = tf.keras.layers.LSTM(self.neurons, return_sequences=True)(lstm_3)
        output_layer = tf.keras.layers.LSTM(self.predictionHorizon)(lstm_4)

        # Create model
        self.model_LSTM = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        return self.model_LSTM

    def time_series_cv(self, model):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        test_scores = np.zeros((self.n_splits, 5, self.y.shape[1]))

        initial_weights = model.get_weights()
        all_fold_weights = {}

        for count, (train_idx, valid_idx) in enumerate(tscv.split(self.X)):
            print(f"Fold {count + 1}")

            X_train, X_valid = self.X[train_idx], self.X[valid_idx]
            y_train, y_valid = self.y[train_idx], self.y[valid_idx]

            model.set_weights(initial_weights)
            model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae', 'mse'])
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_valid, y_valid), verbose=0)

            # Forecasting
            y_pred = model.predict(self.X_test)

            # Save the model weights of the current fold
            all_fold_weights[f'fold_{count + 1}'] = model.get_weights()

            # Results
            for i in range(self.y.shape[1]):
                test_scores[count, 0, i] = round(mean_squared_error(self.y_test[:, i], y_pred[:, i], squared=True), 3)
                test_scores[count, 1, i] = round(mean_squared_error(self.y_test[:, i], y_pred[:, i], squared=False), 3)
                test_scores[count, 2, i] = round(mean_absolute_error(self.y_test[:, i], y_pred[:, i]), 3)
                test_scores[count, 3, i] = round(np.mean(np.abs(self.y_test[:, i] - y_pred[:, i])) * 100, 3)
                test_scores[count, 4, i] = round(r2_score(self.y_test[:, i], y_pred[:, i]), 3)

        return test_scores, all_fold_weights




