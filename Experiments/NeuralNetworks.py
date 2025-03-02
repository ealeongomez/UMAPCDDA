import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tabulate import tabulate
from sklearn.model_selection import TimeSeriesSplit

# Configurar TensorFlow para entrenamiento en múltiples GPUs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Función de pérdida personalizada: Kernel MSE
def kernel_mse_loss(y_true, y_pred):
    sigma = tf.sqrt(2.0) / 2.0
    diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(1.0 - tf.exp(-diff / (2.0 * sigma**2)))

# Función MAPE personalizada con umbral
def calculate_mape(y_true, y_pred, threshold=1e-2):
    y_true_safe = np.where(np.abs(y_true) < threshold, threshold, y_true)
    return mean_absolute_percentage_error(y_true_safe, y_pred) * 100

class ForecastingModels:
    def __init__(self, X, y, X_test, y_test, neurons, num_layers, batch_size, epochs, loss_function, n_splits=5, use_cv=True): 
        self.X = X
        self.y = y
        self.X_test = X_test        
        self.y_test = y_test
        self.neurons = neurons
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs  
        self.n_splits = n_splits
        self.use_cv = use_cv  # Permite activar/desactivar validación cruzada
        self.predictionHorizon = self.y.shape[1] if len(self.y.shape) > 1 else 1

        # Selección de la función de pérdida
        loss_functions = {
            "mse": "mse",
            "mae": "mae",
            "KernelMSE": kernel_mse_loss
        }

        if loss_function not in loss_functions:
            raise ValueError("Invalid loss function. Choose 'mse', 'mae', or 'KernelMSE'")

        self.loss_function = loss_functions[loss_function]

    def Performance_Metrics(self, forecasting_test):
        metrics = {"MSE": [], "RMSE": [], "MAE": [], "MAPE": [], "R2": []}
        col_names = ["Metrics"] + [str(k + 1) for k in range(self.predictionHorizon)]

        for k in range(self.predictionHorizon):
            y_true, y_pred = self.y_test[:, k], forecasting_test[:, k]
            metrics["MSE"].append(round(mean_squared_error(y_true, y_pred), 3))
            metrics["RMSE"].append(round(np.sqrt(mean_squared_error(y_true, y_pred)), 3))
            metrics["MAE"].append(round(mean_absolute_error(y_true, y_pred), 3))
            metrics["MAPE"].append(round(calculate_mape(y_true, y_pred), 3))
            metrics["R2"].append(round(r2_score(y_true, y_pred), 3))

        return tabulate([[k] + v for k, v in metrics.items()], headers=col_names, tablefmt="fancy_grid"), metrics

    def build_and_train_model(self, model_type, X_train, y_train, X_valid, y_valid):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_layer = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]), name='Input')
            layer = input_layer

            for i in range(self.num_layers):
                return_sequences = i < self.num_layers - 1
                RNNLayer = {"RNN": keras.layers.SimpleRNN,
                            "GRU": keras.layers.GRU,
                            "LSTM": keras.layers.LSTM}.get(model_type)

                if not RNNLayer:
                    raise ValueError("Invalid model type")

                layer = RNNLayer(self.neurons, activation='relu', return_sequences=return_sequences, name=f'h{i+1}')(layer)

            output_layer = keras.layers.Dense(self.predictionHorizon, name='Output')(layer)
            model = keras.Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss=self.loss_function, optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae', 'mse'])
            history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_valid, y_valid), verbose=0)
        
        return model, history

    def train_without_cv(self, model_type):
        """Entrena el modelo sin validación cruzada"""
        X_train, X_valid, y_train, y_valid = self.X, self.X_test, self.y, self.y_test
        model, history = self.build_and_train_model(model_type, X_train, y_train, X_valid, y_valid)
        y_pred = model.predict(self.X_test)
        metrics_table, metrics_dict = self.Performance_Metrics(y_pred)
        return metrics_table, metrics_dict, model, history, y_pred

    def time_series_cv(self, model_type):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        test_scores = np.zeros((self.n_splits, 5, self.y.shape[1]))  
        all_history, all_pred = {}, {}

        for count, (train_idx, valid_idx) in enumerate(tscv.split(self.X)):
            print(f"Fold {count + 1}")
            X_train, X_valid = self.X[train_idx], self.X[valid_idx]
            y_train, y_valid = self.y[train_idx], self.y[valid_idx]
            model, history = self.build_and_train_model(model_type, X_train, y_train, X_valid, y_valid)
            y_pred = model.predict(self.X_test)
            all_history[f'fold_{count + 1}'] = history
            all_pred[f'fold_{count + 1}'] = y_pred
            for i in range(self.y.shape[1]):
                y_true_i = self.y_test[:, i]
                y_pred_i = y_pred[:, i]
                test_scores[count, 0, i] = round(mean_squared_error(y_true_i, y_pred_i), 3)
                test_scores[count, 1, i] = round(np.sqrt(mean_squared_error(y_true_i, y_pred_i)), 3)
                test_scores[count, 2, i] = round(mean_absolute_error(y_true_i, y_pred_i), 3)
                test_scores[count, 3, i] = round(calculate_mape(y_true_i, y_pred_i), 3)
                test_scores[count, 4, i] = round(r2_score(y_true_i, y_pred_i), 3)
        return test_scores, all_history, all_pred, None, None

    def train_model(self, model_type):
        return self.time_series_cv(model_type) if self.use_cv else self.train_without_cv(model_type)

    def RNNSimple_Model(self):
        return self.train_model('RNN')

    def GRU_Model(self):
        return self.train_model('GRU')

    def LSTM_Model(self):
        return self.train_model('LSTM')
