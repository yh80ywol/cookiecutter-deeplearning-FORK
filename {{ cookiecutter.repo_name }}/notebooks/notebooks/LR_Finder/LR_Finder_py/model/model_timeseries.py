#UPDATED MODEL.PY WITH LR_FINDER AS CALLBACK!
#UPDATED MODEL.PY WITH LR_FINDER AS CALLBACK!
#UPDATED MODEL.PY WITH LR_FINDER AS CALLBACK!

import sys
sys.path.insert(0,"src/preprocess/")

from model_baseclass import * 
import preprocess

import tensorflow as tf
import hydra, prefect
import matplotlib.pyplot as plt
import mlflow

from prefect import task
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split

#LRFinder Class
class LRFinder(Callback):
    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 1500, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        plt.savefig('reports/LR_Finder_TimeSeries.svg')
        plt.show()

#TimeSeries
class ModelClass_TimeSeries(ModelBaseClass_TimeSeries):
    def __init__(self, config):
        self.batch_size = config.preprocess_timeseries.batch_size

        self.model = config.model_timeseries.model.lower()
        self.num_classes = config.model_timeseries.num_classes
        self.epochs = config.model_timeseries.epochs
        self.learning_rate = config.model_timeseries.learning_rate 
        self.opt = config.model_timeseries.optimizer.lower()
        self.metrics = config.model_timeseries.metrics.lower()
        self.loss = config.model_timeseries.loss.lower()
        self.mlflow_name = config.model_timeseries.mlflow.experiment_name
        self.mlflow_uri = config.model_timeseries.mlflow.tracking_uri
        self.mlflow_log_models = config.model_timeseries.mlflow.log_models
        self.callbacks = config.model_timeseries.callbacks 
    
    def build_model(self, input_shape):
        model = super().build_model(input_shape, self.model, self.num_classes)
        return model

    def compile_model(self, model):
        super().compile_model(model, self.opt, self.loss, self.metrics, self.learning_rate)

    def callbacks_model(self, dataset):
        callbacks = super().callbacks_model(dataset)
        return callbacks

@hydra.main(config_path="../../../../../configs/", config_name="config")
def model_timeseries(cfg):
    #1. Preprocess TimeSeries
    preprocess_class = preprocess.PreprocessingClass_TimeSeries(cfg)
    #1.1 load data
    raw_data = preprocess_class.load_data()
    #1.2 clean data
    data = preprocess_class.cleaning_data(raw_data)
    #1.3 transform data (with sklearn split)
    #1.3.1 split in train/test/val
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_class.sklearn_split(data)
    #1.3.2 feature encoding (standardization or normalization or none)
    if preprocess_class.feature_encoding == "standardization":
        x_train = preprocess_class.sklearn_standardization(x_train)
        x_val = preprocess_class.sklearn_standardization(x_val)
        x_test = preprocess_class.sklearn_standardization(x_test)
    elif preprocess_class.feature_encoding == "normalization":
        x_train = preprocess_class.sklearn_normalization(x_train)
        x_val = preprocess_class.sklearn_normalization(x_val)
        x_test = preprocess_class.sklearn_normalization(x_test) 
    #1.3.3 label encoding (labelencoding string -> int, onehot or none)
    if preprocess_class.label_encoding == "labelencoding":
        y_train = preprocess_class.sklearn_labelencoder(y_train)
        y_val = preprocess_class.sklearn_labelencoder(y_val)    
        y_test = preprocess_class.sklearn_labelencoder(y_test)   
    elif preprocess_class.label_encoding == "onehot":
        y_train = preprocess_class.sklearn_onehot(y_train)
        y_val = preprocess_class.sklearn_onehot(y_val)
        y_test = preprocess_class.sklearn_onehot(y_test)  
    #1.3.4 reshaping features (for fiting into 1D CNN)
    x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.values.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)
    input_shape = x_train.shape[1:]

    #2. Train model
    model_class = ModelClass_TimeSeries(cfg)

    #2.1 build & compile model
    model = model_class.build_model(input_shape)
    model_class.compile_model(model)

    #2.2 initialize callbacks for model
    callbacks = LRFinder() #just LR_Finder

    #2.3 set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=model_class.mlflow_uri)
    mlflow.set_experiment(experiment_name=model_class.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=model_class.mlflow_log_models)

    #2.4 training model
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=model_class.batch_size, epochs=model_class.epochs, callbacks=[callbacks])
    callbacks.plot()
    """
    #2.5 testing model
    score = model.evaluate(x_test, y_test, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    """

if __name__ == "__main__":
    model_timeseries() #tested and worked ("Einpressen_Gesamt_raw")

