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

@hydra.main(config_path="../../configs/", config_name="config")
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
    callbacks = model_class.callbacks_model(x_train) #x_train just needed for Callback SuperConvergence

    #2.3 set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=model_class.mlflow_uri)
    mlflow.set_experiment(experiment_name=model_class.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=model_class.mlflow_log_models)

    #2.4 training model
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=model_class.batch_size, epochs=model_class.epochs, callbacks=callbacks)
    
    #2.5 testing model
    score = model.evaluate(x_test, y_test, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


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

@hydra.main(config_path="../../configs/", config_name="config")
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
    callbacks = model_class.callbacks_model(train_ds) #train_ds just needed for Callback SuperConvergence

    #2.3 set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=model_class.mlflow_uri)
    mlflow.set_experiment(experiment_name=model_class.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=model_class.mlflow_log_models)

    #2.4 train model
    history = model.fit(train_ds, validation_data=val_ds, batch_size=model_class.batch_size, epochs=model_class.epochs, callbacks=callbacks)
    #2.5 testing model
    score = model.evaluate(test_ds, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


#Images 
class ModelClass_Images(ModelBaseClass_Images):
    def __init__(self, config):
        self.batch_size = config.preprocess_images.batch_size
        
        self.model = config.model_images.model.lower()
        self.weights = config.model_images.weights.lower()
        self.train_params = config.model_images.train_all_params
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
        model = super().build_model(augmentation, rescaling, self.model, self.weights, self.num_classes, self.input_shape, self.train_params)
        return model
    
    def compile_model(self, model):
        super().compile_model(model, self.opt, self.loss, self.metrics, self.learning_rate)

    def callbacks_model(self, dataset):
        callbacks = super().callbacks_model(dataset)
        return callbacks

@hydra.main(config_path="../../configs/", config_name="config")
def model_images(cfg):
    #0. Preprocess Images (rescaling and augmentation, used in build_model)
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
    callbacks = model_class.callbacks_model(train_ds) #train_ds just needed for Callback SuperConvergence
    
    #2.3 set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=model_class.mlflow_uri)
    mlflow.set_experiment(experiment_name=model_class.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=model_class.mlflow_log_models)

    #2.4 train model
    history = model.fit(train_ds, validation_data=valid_ds, batch_size=model_class.batch_size, epochs=model_class.epochs, callbacks=callbacks)
    
    #2.5 testing model
    score = model.evaluate(test_ds, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    

if __name__ == "__main__":
    model_timeseries() #tested and worked ("Einpressen_Gesamt_raw")
    #model_structureddata() #tested and worked (HeartDisease, PetFinderMini)
    #model_images() #tested and worked (Fashion MNIST, cats/dogs)








### Tasks for Prefect UI ###
#-> tasks has to be outside of any class
#-> class_input equals to self, but not possible due to fact that tasks cannot be in classes
#-> nout=2 cause 2 return values(e.g. images_train/images_val), that causes in output load_data[0]/[1]
#-> checkpoint=False very important by using other frameworks(!), otherwise prefect and tf/keras/sklearn wont work in prefect agent/server

#TimeSeries Tasks
@task(name="train_model", checkpoint=False, nout=2)
def timeseries_train_model_task(class_input, x_train, y_train, x_val, y_val):
    logger = prefect.context.get("logger")
    logger.info("Training model on Train/Valdata...")

    input_shape = x_train.shape[1:]
    #build model
    model = class_input.build_model(input_shape)
    #compile model
    class_input.compile_model(model)
    #initialize callbacks
    callbacks = class_input.callbacks_model(x_train)
    #set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=class_input.mlflow_uri)
    mlflow.set_experiment(experiment_name=class_input.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=class_input.mlflow_log_models)
    #training model
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=class_input.batch_size, epochs=class_input.epochs, callbacks=callbacks)

    return model

@task(name="test_model", checkpoint=False)
def timeseries_test_model_task(class_input, trained_model, x_test, y_test):
    logger = prefect.context.get("logger")
    logger.info("Testing model on Testdata...")

    #testing model
    score = trained_model.evaluate(x_test, y_test, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


#Structured Data Tasks
@task(name="train_model", checkpoint=False, nout=2)
def structureddata_train_model_task(class_input, train_ds, val_ds, all_inputs, encoded_features):
    logger = prefect.context.get("logger")
    logger.info("Training model on Train/Valdata...")

    #build model
    model = class_input.build_model(all_inputs, encoded_features)
    #compile model
    class_input.compile_model(model)
    #initialize callbacks
    callbacks = class_input.callbacks_model(train_ds)
    #set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=class_input.mlflow_uri)
    mlflow.set_experiment(experiment_name=class_input.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=class_input.mlflow_log_models)
    #training model
    history = model.fit(train_ds, validation_data=val_ds, epochs=class_input.epochs, callbacks=callbacks)

    return model

@task(name="test_model", checkpoint=False)
def structureddata_test_model_task(class_input, trained_model, test_ds):
    logger = prefect.context.get("logger")
    logger.info("Testing model on Testdata...")
    
    #testing model
    score = trained_model.evaluate(test_ds, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


#Images Tasks
@task(name="train_model", checkpoint=False, nout=2)
def images_train_model_task(class_input, train_ds, valid_ds, augmentation, rescaling):
    logger = prefect.context.get("logger")
    logger.info("Training model on Train/Valdata...")

    #build model
    model = class_input.build_model(augmentation, rescaling)
    #compile model
    class_input.compile_model(model)
    #initialize callbacks
    callbacks = class_input.callbacks_model(train_ds)
    #set mlflow for tracking metrics and params
    mlflow.set_tracking_uri(uri=class_input.mlflow_uri)
    mlflow.set_experiment(experiment_name=class_input.mlflow_name)
    mlflow.start_run()
    mlflow.autolog(log_models=class_input.mlflow_log_models)
    #train model
    history = model.fit(train_ds, validation_data=valid_ds, batch_size=class_input.batch_size, epochs=class_input.epochs, callbacks=callbacks)
   
    return model

@task(name="test_model", checkpoint=False)
def images_test_model_task(class_input, trained_model, test_ds):
    logger = prefect.context.get("logger")
    logger.info("Testing model on Testdata...")
    
    #testing model
    score = trained_model.evaluate(test_ds, verbose=1)
    mlflow.log_metrics({"test_loss": score[0], "test_acc": score[1]})
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')