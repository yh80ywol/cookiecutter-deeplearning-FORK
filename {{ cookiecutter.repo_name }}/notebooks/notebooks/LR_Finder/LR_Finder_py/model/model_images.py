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
        plt.savefig('reports/LR_Finder_Images.svg')
        plt.show()

#Images 
class ModelClass_Images(ModelBaseClass_Images):
    def __init__(self, config):
        self.batch_size = config.preprocess_images.batch_size
        
        self.model = config.model_images.model.lower()
        self.weights = config.model_images.weights.lower()
        self.num_classes = config.model_images.num_classes
        self.input_shape = config.preprocess_images.image_size + config.preprocess_images.color_channels
        self.epochs = config.model_images.epochs
        self.opt = config.model_images.optimizer.lower()
        self.metrics = config.model_images.metrics.lower()
        self.loss = config.model_images.loss.lower()
        self.learning_rate = config.model_images.learning_rate
        self.mlflow_name = config.model_images.mlflow.experiment_name
        self.mlflow_uri = config.model_images.mlflow.tracking_uri
        self.mlflow_log_models = config.model_images.mlflow.log_models
        self.callbacks = config.model_images.callbacks

    def build_model(self, augmentation, rescaling):
        model = super().build_model(augmentation, rescaling, self.model, self.weights, self.num_classes, self.input_shape)
        return model
    
    def compile_model(self, model):
        super().compile_model(model, self.opt, self.loss, self.metrics, self.learning_rate)

    def callbacks_model(self, dataset):
        callbacks = super().callbacks_model(dataset)
        return callbacks

@hydra.main(config_path="../../../../../configs/", config_name="config")
def model_images(cfg):
    #Preprocess Images (rescaling and augmentation, used in build_model)
    preprocess_class = preprocess.PreprocessingClass_Images(cfg)
    rescaling = preprocess_class.rescaling_images()
    augmentation = preprocess_class.augmenting_images()

    #1. load images / get dataset
    #1.1 own dataset (see in path in config.yaml)
    train_ds, valid_ds, test_ds = preprocess_class.load_images()

    #1.2 fashion mnist (for testing and proove)
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()
    x_train = x_train[..., tf.newaxis]
    x_valid = x_valid[..., tf.newaxis]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)
    """

    #2. model
    model_class = ModelClass_Images(cfg)
	
    #2.1 build & compile model
    model = model_class.build_model(augmentation,rescaling)
    model_class.compile_model(model)

    #2.2 initialize callbacks for model
    callbacks = LRFinder() #just LR_Finder
    
    #2.3 set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=model_class.mlflow_uri)
    mlflow.set_experiment(experiment_name=model_class.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=model_class.mlflow_log_models)

    #2.4 train model
    history = model.fit(train_ds, validation_data=valid_ds, batch_size=model_class.batch_size, epochs=model_class.epochs, callbacks=[callbacks])
    callbacks.plot()
    """
    #2.5 testing model
    score = model.evaluate(test_ds, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    """

if __name__ == "__main__":
    model_images() #tested and worked (Fashion MNIST, cats/dogs)
