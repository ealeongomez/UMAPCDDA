import os, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
from sklearn.model_selection import train_test_split

# Configurar TensorFlow para usar múltiples GPUs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

strategy = tf.distribute.MirroredStrategy()

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def kernel_mse_loss(y_true, y_pred):
    sigma = tf.sqrt(2.0) / 2.0
    diff = tf.square(y_true - y_pred)
    loss = 1.0 - tf.exp(-diff / (2.0 * sigma**2))
    return tf.reduce_mean(loss)

class ForecastingModels:
    def __init__(self, X_train, X_test, X_valid, y_train, y_valid, y_test, neurons, batch_size, epochs, loss_function, num_layers=4):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.neurons = neurons
        self.batch_size = batch_size
        self.epochs = epochs  
        self.num_layers = num_layers
        self.predictionHorizon = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1

        # Seleccionar la función de pérdida según el string proporcionado
        if loss_function == 'mse':
            self.loss_function = 'mse'
        elif loss_function == 'mae':
            self.loss_function = 'mae'
        elif loss_function == 'kernel':
            self.loss_function = kernel_mse_loss
        else:
            raise ValueError("Invalid loss function. Choose 'mse', 'mae', or 'kernel'")

    def Performance_Metrics(self, forecasting_test):
        metrics = {"MSE": [], "RMSE": [], "MAE": [], "MAPE": [], "R2": []}
        col_names = ["Metrics"] + [str(k + 1) for k in range(self.predictionHorizon)]

        for k in range(self.predictionHorizon):
            y_true, y_pred = self.y_test[:, k], forecasting_test[:, k]
            metrics["MSE"].append(round(mean_squared_error(y_true, y_pred), 3))
            metrics["RMSE"].append(round(mean_squared_error(y_true, y_pred) ** 0.5, 3))
            metrics["MAE"].append(round(mean_absolute_error(y_true, y_pred), 3))
            metrics["MAPE"].append(np.mean(np.abs(y_true - y_pred)) * 100)
            metrics["R2"].append(round(r2_score(y_true, y_pred), 3))

        data = [[key] + value for key, value in metrics.items()]
        return tabulate(data, headers=col_names, tablefmt="fancy_grid"), metrics

    def build_and_train_model(self, model_type):
        with strategy.scope():
            input_layer = keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]), name='Input')
            layer = input_layer

            for i in range(self.num_layers):
                return_sequences = i < self.num_layers - 1  # La última capa no devuelve secuencias
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
            model.compile(loss=self.loss_function, optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae', 'mse'])

            history = model.fit(
                self.X_train, self.y_train, 
                epochs=self.epochs, batch_size=self.batch_size, 
                validation_data=(self.X_valid, self.y_valid), verbose=0
            )

        forecasting = model.predict(self.X_test)
        table, metrics = self.Performance_Metrics(forecasting)
        return model, pd.DataFrame(history.history), table, metrics, forecasting

    def RNNSimple_Model(self):
        return self.build_and_train_model('RNN')

    def GRU_Model(self):
        return self.build_and_train_model('GRU')

    def LSTM_Model(self):
        return self.build_and_train_model('LSTM')
