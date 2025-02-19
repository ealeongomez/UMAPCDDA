import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tabulate import tabulate
from sklearn.model_selection import TimeSeriesSplit

# Configure TensorFlow for multi-GPU training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
strategy = tf.distribute.MirroredStrategy()

# Custom loss function: Kernel MSE
def kernel_mse_loss(y_true, y_pred):
    sigma = tf.sqrt(2.0) / 2.0
    diff = tf.square(y_true - y_pred)
    loss = 1.0 - tf.exp(-diff / (2.0 * sigma**2))
    return tf.reduce_mean(loss)

# Custom MAPE function with threshold handling
def calculate_mape(y_true, y_pred, threshold=1e-2):
    if y_true.shape != y_pred.shape:
        raise ValueError("The dimensions of y_true and y_pred must match")
    
    y_true_safe = np.where(np.abs(y_true) < threshold, threshold, y_true)
    mape = mean_absolute_percentage_error(y_true_safe, y_pred) * 100
    return mape

class ForecastingModels:
    def __init__(self, X, y, X_test, y_test, neurons, num_layers, batch_size, epochs, loss_function, n_splits=5): 
        self.X = X
        self.y = y
        self.X_test = X_test        
        self.y_test = y_test
        self.neurons = neurons
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs  
        self.n_splits = n_splits
        self.predictionHorizon = self.y.shape[1] if len(self.y.shape) > 1 else 1

        # Select loss function
        if loss_function == 'mse':
            self.loss_function = 'mse'
        elif loss_function == 'mae':
            self.loss_function = 'mae'
        elif loss_function == 'KernelMSE':
            self.loss_function = kernel_mse_loss
        else:
            raise ValueError("Invalid loss function. Choose 'mse', 'mae', or 'KernelMSE'")

    def Performance_Metrics(self, forecasting_test):
        metrics = {"MSE": [], "RMSE": [], "MAE": [], "MAPE": [], "R2": []}
        col_names = ["Metrics"] + [str(k + 1) for k in range(self.predictionHorizon)]

        for k in range(self.predictionHorizon):
            y_true, y_pred = self.y_test[:, k], forecasting_test[:, k]
            metrics["MSE"].append(round(mean_squared_error(y_true, y_pred), 3))
            metrics["RMSE"].append(round(mean_squared_error(y_true, y_pred) ** 0.5, 3))
            metrics["MAE"].append(round(mean_absolute_error(y_true, y_pred), 3))
            metrics["MAPE"].append(round(calculate_mape(y_true, y_pred), 3))
            metrics["R2"].append(round(r2_score(y_true, y_pred), 3))

        data = [[key] + value for key, value in metrics.items()]
        return tabulate(data, headers=col_names, tablefmt="fancy_grid"), metrics

    def build_and_train_model(self, model_type, X_train, y_train, X_valid, y_valid):
        with strategy.scope():
            input_layer = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]), name='Input')
            layer = input_layer

            for i in range(self.num_layers):
                return_sequences = i < self.num_layers - 1
                if model_type == 'RNN':
                    layer = keras.layers.SimpleRNN(self.neurons, activation='relu', return_sequences=return_sequences, name=f'h{i+1}')(layer)
                elif model_type == 'GRU':
                    layer = keras.layers.GRU(self.neurons, activation='relu', return_sequences=return_sequences, name=f'h{i+1}')(layer)
                elif model_type == 'LSTM':
                    layer = keras.layers.LSTM(self.neurons, activation='relu', return_sequences=return_sequences, name=f'h{i+1}')(layer)
                else:
                    raise ValueError("Invalid model type")

            output_layer = keras.layers.Dense(self.predictionHorizon, name='Output')(layer)
            model = keras.Model(inputs=input_layer, outputs=output_layer)

            #print(model.summary())
            model.compile(loss=self.loss_function, optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae', 'mse'])

            history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_valid, y_valid), verbose=0)
        
        return model, history

    def time_series_cv(self, model_type):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        test_scores = np.zeros((self.n_splits, 5, self.y.shape[1]))  # Store metrics for each fold
        all_history, all_pred = {}, {}, {} 

        for count, (train_idx, valid_idx) in enumerate(tscv.split(self.X)):
            print(f"Fold {count + 1}")

            X_train, X_valid = self.X[train_idx], self.X[valid_idx]
            y_train, y_valid = self.y[train_idx], self.y[valid_idx]

            model, history = self.build_and_train_model(model_type, X_train, y_train, X_valid, y_valid)
            y_pred = model.predict(self.X_test)

            # Save 
            all_history[f'fold_{count + 1}'] = history
            all_pred[f'fold_{count + 1}'] = y_pred

            # Store metrics for each fold
            for i in range(self.y.shape[1]):
                test_scores[count, 0, i] = round(mean_squared_error(self.y_test[:, i], y_pred[:, i]), 3)
                test_scores[count, 1, i] = round(mean_squared_error(self.y_test[:, i], y_pred[:, i]) ** 0.5, 3)
                test_scores[count, 2, i] = round(mean_absolute_error(self.y_test[:, i], y_pred[:, i]), 3)
                test_scores[count, 3, i] = round(np.mean(np.abs(self.y_test[:, i] - y_pred[:, i])) * 100, 3)
                test_scores[count, 4, i] = round(r2_score(self.y_test[:, i], y_pred[:, i]), 3)

        return test_scores, all_history, all_pred

    def RNNSimple_Model(self):
        test_scores, all_fold_weights, all_history, all_pred = self.time_series_cv('RNN') 
        return test_scores, all_history, all_pred

    def GRU_Model(self):
        test_scores, all_fold_weights, all_history, all_pred = self.time_series_cv('GRU')
        return test_scores, all_history, all_pred

    def LSTM_Model(self):
        test_scores, all_fold_weights, all_history, all_pred = self.time_series_cv('LSTM')
        return test_scores, all_history, all_pred
