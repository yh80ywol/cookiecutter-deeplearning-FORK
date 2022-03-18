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
        plt.savefig('reports/LR_Finder_StructuredData.svg')
        plt.show()

#StructuredData
class ModelClass_StructuredData(ModelBaseClass_StructuredData):
    def __init__(self, config):
        self.batch_size = config.preprocess_structureddata.batch_size

        self.model = config.model_structureddata.model.lower()
        self.num_classes = config.model_structureddata.num_classes
        self.epochs = config.model_structureddata.epochs
        self.learning_rate = config.model_structureddata.learning_rate
        self.opt = config.model_structureddata.optimizer.lower()
        self.metrics = config.model_structureddata.metrics.lower()
        self.loss = config.model_structureddata.loss.lower()
        self.mlflow_name = config.model_structureddata.mlflow.experiment_name
        self.mlflow_uri = config.model_structureddata.mlflow.tracking_uri
        self.mlflow_log_models = config.model_structureddata.mlflow.log_models
        self.callbacks = config.model_structureddata.callbacks 
    
    def build_model(self, all_inputs, encoded_features):
        model = super().build_model(all_inputs, encoded_features, self.model, self.num_classes)
        return model

    def compile_model(self, model):
        super().compile_model(model, self.opt, self.loss, self.metrics, self.learning_rate)
    
    def callbacks_model(self, dataset):
        callbacks = super().callbacks_model(dataset)
        return callbacks

@hydra.main(config_path="../../../../../configs/", config_name="config")
def model_structureddata(cfg):
    #Preprocess StructuredData
    preprocess_class = preprocess.PreprocessingClass_StructuredData(cfg)
    
    #1. load/clean data
    #1.1 own dataset
    """
        raw_data = preprocess_class.load_data()
        data = preprocess_class.cleaning_data(raw_data)
        train_ds, val_ds, test_ds = preprocess_class.tf_dataframe_to_dataset(data)
        [(train_features, label_batch)] = train_ds.take(1)
        print('Every feature:', list(train_features.keys()))
        print('A batch of features:', train_features["200"])
        print('A batch of targets:', label_batch)  
        #1.1.1 set inputs/features
        all_inputs = []
        encoded_features = []
        preprocess_class.tf_set_normalization(preprocess_class.numeric_columns, train_ds, all_inputs, encoded_features)
        preprocess_class.tf_set_category_to_onehot(preprocess_class.int_to_onehot_columns, train_ds, all_inputs, encoded_features, "int64", 3)
        preprocess_class.tf_set_category_to_onehot(preprocess_class.str_to_onehot_columns, train_ds, all_inputs, encoded_features, "string", 3)
        preprocess_class.tf_set_category_to_integer(preprocess_class.str_to_int_columns, train_ds, all_inputs, encoded_features)
    """

    #1.2 Official Datasets (for testing and proove)
    #1.2.1 Heart Disease (~80% val. acc. 50 epochs, https://keras.io/examples/structured_data/structured_data_classification_from_scratch/)
    #get dataset
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        #df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    dataframe = pd.read_csv(file_url)
    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)
    train_ds = df_to_dataset(train_dataframe, batch_size=16)
    val_ds = df_to_dataset(val_dataframe, shuffle=True, batch_size=16)
    test_ds = val_ds.take(int(len(val_ds)*0.5))
    val_ds = val_ds.skip(int(len(val_ds)*0.5))
    #set features
    all_inputs = []
    encoded_features = []
    preprocess_class.tf_set_normalization(["age","trestbps","chol","thalach","oldpeak","slope"], train_ds, all_inputs, encoded_features)
    preprocess_class.tf_set_category_to_onehot(["sex","cp","fbs","restecg","exang","ca"], train_ds, all_inputs, encoded_features, "int64", 3)
    preprocess_class.tf_set_category_to_onehot(["thal"], train_ds, all_inputs, encoded_features, "string", 3)

    
    #1.2.2 PetFinder (~75% val.acc., 10 Epochs, https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers)
    """
        dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
        csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
        tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')
        dataframe = pd.read_csv(csv_file)
        # In the original dataset "4" indicates the pet was not adopted.
        dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)
        #drop un-used columns.
        dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
        #split
        train, test = train_test_split(dataframe, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)
        print(len(train), 'train examples')
        print(len(val), 'validation examples')
        print(len(test), 'test examples')
        train_ds = df_to_dataset(train, batch_size=16)
        val_ds = df_to_dataset(val, shuffle=False, batch_size=16)
        test_ds = df_to_dataset(test, shuffle=False, batch_size=16)
        #set features
        all_inputs = []
        encoded_features = []
        preprocess_class.tf_set_normalization(["PhotoAmt","Fee"], train_ds, all_inputs, encoded_features)
        preprocess_class.tf_set_category_to_onehot(["Age"], train_ds, all_inputs, encoded_features, "int64", 5)
        preprocess_class.tf_set_category_to_onehot(["Type","Color1","Color2","Gender","MaturitySize","FurLength",
                        "Vaccinated","Sterilized","Health","Breed1"], train_ds, all_inputs, encoded_features, "string", 5)
    """


    #2. model
    model_class = ModelClass_StructuredData(cfg)

    #2.1 build & compile model
    model = model_class.build_model(all_inputs, encoded_features)
    model_class.compile_model(model)

    #2.2 initialize callbacks for model
    callbacks = LRFinder() #just LR_Finder

    #2.3 set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=model_class.mlflow_uri)
    mlflow.set_experiment(experiment_name=model_class.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=model_class.mlflow_log_models)

    #2.4 train model
    history = model.fit(train_ds, validation_data=val_ds, batch_size=model_class.batch_size, epochs=model_class.epochs, callbacks=[callbacks])
    callbacks.plot()
    """
    #2.5 testing model
    score = model.evaluate(test_ds, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    """


if __name__ == "__main__":
    model_structureddata() #tested and worked (HeartDisease, PetFinderMini)

