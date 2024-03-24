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
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================================================================
# Class
# ==================================================================
 
class ForecastingModels():                              

    def __init__(self, X_train, X_test, X_valid, y_train, y_valid, y_test, neurons, batch_size, epochs): 
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.neurons = neurons
        self.batch_size = batch_size
        self.epochs = epochs  
        try:
            self.predictionHorizon = self.y_train.shape[1] 
        except:
            self.predictionHorizon = 1

    def mean_absolute_percentage_error(self, forecasting_test):
        mape = np.mean(np.abs((self.y_test - forecasting_test)/self.y_test))        
        return mape

    """
        Performance metrics (MSE, RMSE, MAE, R2) ---------------------------------------------------------------------
    """
    def Performance_Metrics(self, forecasting_test):
        col_names, MSE, RMSE, MAE, MAPE, R2 = ['Metrics'], ["MSE"], ["RMSE"], ["MAE"], ["MAPE"], ["R2"]
        for k in range(self.predictionHorizon):
            col_names.append(str(k+1))
            MSE.append(round(mean_squared_error(self.y_test[:,k], forecasting_test[:,k], squared=False), 3))
            RMSE.append(round(mean_squared_error(self.y_test[:,k], forecasting_test[:,k], squared=True), 3))
            MAE.append(round(mean_absolute_error(self.y_test[:,k], forecasting_test[:,k]), 3))
            MAPE.append(np.mean(np.abs(self.y_test[:,k] - forecasting_test[:,k]))*100)
            R2.append(round(r2_score(self.y_test[:,k], forecasting_test[:,k]), 3))

        self.data = [MSE, RMSE, MAE, MAPE, R2]
        metricsDictionary = {"MSE": MSE, "RMSE": RMSE, "MAE": MAE, "MAPE": MAPE, "R2": R2}

        return tabulate(self.data, headers=col_names, tablefmt="fancy_grid"), metricsDictionary

    """
        Recurrent Neural Network Simple ------------------------------------------------------------------------------
    """
    def RNNSimple_Model(self):      
      # Arquitecture 
      input = tf.keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]), name='Input')
      rnn_1 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h1')(input)
      rnn_2 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h2')(rnn_1)
      rnn_3 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h3')(rnn_2)
      rnn_4 = tf.keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=True, name='h4')(rnn_3)
      output = keras.layers.SimpleRNN(self.predictionHorizon)(rnn_4)

      # Create model
      self.model_RNNSimple = tf.keras.Model(inputs=input, outputs=output)

      # Compile model
      self.model_RNNSimple.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae', 'mse'])
      self.history_RNNSimple = self.model_RNNSimple.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_valid, self.y_valid), verbose=0)
      
      self.forecasting_RNNSimple = self.model_RNNSimple.predict(self.X_test)
      self.table_RNNSimple, self.metrics_RNNSimple = ForecastingModels.Performance_Metrics(self, self.forecasting_RNNSimple)

      return self.model_RNNSimple, pd.DataFrame(self.history_RNNSimple.history), self.table_RNNSimple, self.metrics_RNNSimple, self.forecasting_RNNSimple
       
    """
        Gate Recurrent Unit ------------------------------------------------------------------------------------------
    """
    def GRU_Model(self):

        input = tf.keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]), name='Input')
        gru_1 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h1')(input)
        gru_2 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h2')(gru_1)
        gru_3 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h3')(gru_2)
        gru_4 = tf.keras.layers.GRU(self.neurons, activation='relu', return_sequences=True, name='h4')(gru_3)
        output = keras.layers.GRU(self.predictionHorizon)(gru_4)
        self.model_GRU = tf.keras.Model(inputs=input, outputs=output)

        self.model_GRU.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae', 'mse'])
        self.history_GRU = self.model_GRU.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_valid, self.y_valid), verbose=0)

        self.forecasting_GRU = self.model_GRU.predict(self.X_test)
        self.table_GRU, self.metrics_GRU = ForecastingModels.Performance_Metrics(self, self.forecasting_GRU)

        return self.model_GRU, pd.DataFrame(self.history_GRU.history), self.table_GRU, self.metrics_GRU, self.forecasting_GRU

    """
        Long Short Temporaly Memory ----------------------------------------------------------------------------------
    """
    def LSTM_Model(self):
        
        input = tf.keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]), name='Input')
        lstm_1 = tf.keras.layers.LSTM(self.neurons, activation='relu', return_sequences=True, name='h1')(input)
        lstm_2 = tf.keras.layers.LSTM(self.neurons, activation='relu', return_sequences=True, name='h2')(lstm_1)
        lstm_3 = tf.keras.layers.LSTM(self.neurons, activation='relu', return_sequences=True, name='h3')(lstm_2)
        lstm_4 = tf.keras.layers.LSTM(self.neurons, activation='relu', return_sequences=True, name='h4')(lstm_3)
        output = keras.layers.LSTM(self.predictionHorizon)(lstm_4)
        self.model_LSTM = tf.keras.Model(inputs=input, outputs=output)

        self.model_LSTM.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae', 'mse'])
        self.history_LSTM = self.model_LSTM.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_valid, self.y_valid), verbose=0)

        self.forecasting_LSTM = self.model_LSTM.predict(self.X_test)
        self.table_LSTM, self.metrics_LSTM = ForecastingModels.Performance_Metrics(self, self.forecasting_LSTM)

        return self.model_LSTM, pd.DataFrame(self.history_LSTM.history), self.table_LSTM, self.metrics_LSTM, self.forecasting_LSTM
