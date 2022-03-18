from preprocess_baseclass import *
import pandas as pd
import numpy as np
import hydra, prefect
from prefect import task

#TimeSeries
class PreprocessingClass_TimeSeries(PreprocessingBaseClass_TimeSeries):
    def __init__(self, config):
        self.file_path = config.preprocess_timeseries.file_path
        self.file_name = config.preprocess_timeseries.file_name
        self.file_sheet = config.preprocess_timeseries.file_sheet
        self.file_columns = config.preprocess_timeseries.file_columns
        self.file_columns_method = config.preprocess_timeseries.file_columns_method

        self.label =  config.preprocess_timeseries.label
        self.feature_encoding = config.preprocess_timeseries.feature_encoding.lower()
        self.label_encoding = config.preprocess_timeseries.label_encoding.lower()
        self.shuffle = config.preprocess_timeseries.shuffle
        self.train_split = config.preprocess_timeseries.split[0]
        self.val_split = config.preprocess_timeseries.split[1]
        self.test_split = config.preprocess_timeseries.split[2]

    def load_data(self):
        data = super().load_data(self.file_path, self.file_name, self.file_sheet)
        return data
    
    def cleaning_data(self, data):
        data = super().cleaning_data(data, self.file_columns, self.file_columns_method, self.label)
        return data

    def sklearn_normalization(self, data): 
        data = super().sklearn_normalization(data)
        return data

    def sklearn_standardization(self, data): 
        data = super().sklearn_standardization(data)
        return data
       
    def sklearn_onehot(self, data, inverse=False):
        data = super().sklearn_onehot(data)
        return data

    def sklearn_labelencoder(self, data):
        data = super().sklearn_labelencoder(data)
        return data

    def sklearn_split(self, data):
        x_train, y_train, x_test, y_test, x_val, y_val = super().sklearn_split(data, self.train_split, self.val_split, self.test_split, self.shuffle)
        return  x_train, y_train, x_test, y_test, x_val, y_val

@hydra.main(config_path="../../configs/", config_name="config")
def main_timeseries(cfg): 
    preprocess = PreprocessingClass_TimeSeries(cfg)
    #1. load data
    raw_data = preprocess.load_data()

    #2. clean data
    data = preprocess.cleaning_data(raw_data)

    #3. transform data (with sklearn split)
    #3.1 split in train/test/val
    x_train, y_train, x_val, y_val, x_test, y_test,  = preprocess.sklearn_split(data)

    #3.2 feature encoding (standardization or normalization or none)
    if preprocess.feature_encoding == "standardization":
        x_train = preprocess.sklearn_standardization(x_train)
        x_val = preprocess.sklearn_standardization(x_val)
        x_test = preprocess.sklearn_standardization(x_test)
    elif preprocess.feature_encoding == "normalization":
        x_train = preprocess.sklearn_normalization(x_train)
        x_val = preprocess.sklearn_normalization(x_val)
        x_test = preprocess.sklearn_normalization(x_test)   

    #3.3 label encoding (labelencoding string -> int, onehot or none)
    if preprocess.label_encoding == "labelencoding":
        y_train = preprocess.sklearn_labelencoder(y_train)
        y_val = preprocess.sklearn_labelencoder(y_val)    
        y_test = preprocess.sklearn_labelencoder(y_test)   
    elif preprocess.label_encoding == "onehot":
        y_train = preprocess.sklearn_onehot(y_train)
        y_val = preprocess.sklearn_onehot(y_val)
        y_test = preprocess.sklearn_onehot(y_test)

    print("x_train (some features):\n", x_train.describe())
    print("x_train (some features):\n", x_val.describe())
    print("x_train (some features):\n", x_test.describe())
    print("y_train (some targets):\n", y_train)

    #3.4 reshaping features (for fiting into 1D CNN)
    x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.values.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)


#StructuredData
class PreprocessingClass_StructuredData(PreprocessingBaseClass_StructuredData):
    def __init__(self, config):
        self.file_path = config.preprocess_structureddata.file_path
        self.file_name = config.preprocess_structureddata.file_name
        self.file_sheet = config.preprocess_structureddata.file_sheet
        self.file_columns = config.preprocess_structureddata.file_columns
        self.file_columns_method = config.preprocess_structureddata.file_columns_method

        self.label =  config.preprocess_structureddata.label
        self.numeric_columns = list(map(str,config.preprocess_structureddata.numeric_columns))
        self.int_to_onehot_columns =  list(map(str,config.preprocess_structureddata.int_to_onehot_columns))
        self.str_to_onehot_columns =  list(map(str,config.preprocess_structureddata.str_to_onehot_columns))
        self.str_to_int_columns =  list(map(str,config.preprocess_structureddata.str_to_int_columns))
        
        self.batch_size = config.preprocess_structureddata.batch_size
        self.shuffle = config.preprocess_structureddata.shuffle
        self.train_split = config.preprocess_structureddata.split[0]
        self.val_split = config.preprocess_structureddata.split[1]
        self.test_split = config.preprocess_structureddata.split[2]

    def load_data(self):
        data = super().load_data(self.file_path, self.file_name, self.file_sheet)
        return data

    def cleaning_data(self, data):
        data = super().cleaning_data(data, self.file_columns, self.file_columns_method, self.label)
        return data
        
    def sklearn_normalization(self, data): 
        data = super().sklearn_normalization(data)
        return data

    def sklearn_standardization(self, data): 
        data = super().sklearn_standardization(data)
        return data
       
    def sklearn_onehot(self, data, inverse=False):
        data = super().sklearn_onehot(data)
        return data

    def tf_layer_normalization(self, name, dataset):
        layer = super().tf_layer_normalization(name, dataset)
        return layer

    def tf_set_normalization(self, columns, dataset, all_inputs, encoded_features):
        super().tf_set_normalization(columns, dataset, all_inputs, encoded_features)

    def tf_layer_category_to_onehot(self, name, dataset, dtype, max_tokens=None):
        layer = super().tf_layer_category_to_onehot(name, dataset, dtype, max_tokens)
        return layer

    def tf_set_category_to_onehot(self, columns, dataset, all_inputs, encoded_features, dtype, max_tokens=None):
        super().tf_set_category_to_onehot(columns, dataset, all_inputs, encoded_features, dtype, max_tokens)

    def tf_layer_category_to_integer(self, name, dataset, max_tokens=None):
        layer = super().tf_layer_category_to_integer(name, dataset, max_tokens)
        return layer

    def tf_set_category_to_integer(self, columns, dataset, all_inputs, encoded_features, max_tokens=None):
        super().tf_set_category_to_integer(columns, dataset, all_inputs, encoded_features, max_tokens)

    def tf_dataframe_to_dataset(self, dataframe):
        train_dataset, val_dataset, test_dataset = super().tf_dataframe_to_dataset(dataframe, self.label, self.train_split, self.val_split, self.test_split, self.batch_size, self.shuffle)
        return train_dataset, val_dataset, test_dataset

@hydra.main(config_path="../../configs/", config_name="config")
def main_structureddata(cfg):  
    preprocess = PreprocessingClass_StructuredData(cfg)
    #1. load data
    raw_data = preprocess.load_data()
    
    #2. clean data
    data = preprocess.cleaning_data(raw_data)

    #3. transform data
    #3.1 sklearn with data (!) (for testing used "Einpressen_Gesamt_Test.xlsx")
    #3.1.1 normalization
        #normalized_data = preprocess.sklearn_normalization(data)
        #print(normalized_data.describe())

    #3.1.2 standardization
        #standardized_data = preprocess.sklearn_standardization(data)
        #print(standardized_data.describe())

    #3.1.3 onehot encoding
        #onehot_data = preprocess.sklearn_onehot(data,inverse=False)
        #print(onehot_data)
        #onehot_data_inverse = preprocess.sklearn_onehot(data,inverse=True)
        #print(onehot_data_inverse)

    #4.2 tensorflow/keras with tf.datasets (!) (for testing used "Einpressen_Gesamt_Test.xlsx")
    #4.2.1 transform formate "dataframe" to "tf.dataset"
    train_ds, val_ds, test_ds = preprocess.tf_dataframe_to_dataset(data)
    [(features, label_batch)] = train_ds.take(1)
    print('Every feature:', list(features.keys()))
    print('A batch of features (200):', features["200"])
    print('A batch of targets:', label_batch)

    #4.2.2 numeric features (standardization/normalization, needs -> name of encoding column, dataset)
    data = features["200"]
    layer = preprocess.tf_layer_normalization("200", train_ds)
    print("raw:", data)
    print("normalized:", layer(data))
    
    #4.2.3 categorical features (nominal, onehot encoding, needs -> name of encoding column, dataset, type, max_tokens)
    #if type=int64 -> integer to onehot /// if type=string -> string to onehot
    data = features["age"]
    layer = preprocess.tf_layer_category_to_onehot("age", train_ds, "int64", 3)
    print("integer:", data)
    print("onehot:", layer(data))

    data = features["size"]
    layer = preprocess.tf_layer_category_to_onehot("size", train_ds, "string", 3)
    print("string:", data)
    print("onehot:", layer(data))

    #4.2.4 categorical features (ordinal, needs -> name of encoding column, dataset)
    #string to integer
    data = features["animal"]
    layer = preprocess.tf_layer_category_to_integer('animal', train_ds)
    print("string:", data)
    print("integer:", layer(data))
    
    #4.2.5 set ALL features for Dataset
    all_inputs = []
    encoded_features = []
    preprocess.tf_set_normalization(["200"], train_ds, all_inputs, encoded_features)
    preprocess.tf_set_category_to_onehot(["age"], train_ds, all_inputs, encoded_features, "int64", 3)
    preprocess.tf_set_category_to_onehot(["size"], train_ds, all_inputs, encoded_features, "string", 3)
    preprocess.tf_set_category_to_integer(["animal"], train_ds, all_inputs, encoded_features)
    print("Input tensors:", all_inputs)
    print("Encoded tensors:", encoded_features)


#Images 
class PreprocessingClass_Images(PreprocessingBaseClass_Images):
    def __init__(self, config):
        #load
        self.file_path = config.preprocess_images.file_path
        self.image_size = config.preprocess_images.image_size
        self.batch_size = config.preprocess_images.batch_size
        self.split = config.preprocess_images.split
        self.seed = config.preprocess_images.seed

        #data augmentation 
        self.rescaling = config.preprocess_images.rescaling
        self.random_crop = config.preprocess_images.random_crop
        self.random_flip = config.preprocess_images.random_flip
        self.random_translation = config.preprocess_images.random_translation
        self.random_rotation = config.preprocess_images.random_rotation
        self.random_zoom = config.preprocess_images.random_zoom

    def load_images(self):
        train_ds, val_ds, test_ds = super().load_images(self.file_path, self.image_size, self.batch_size, self.split, self.seed)
        return train_ds, val_ds, test_ds

    def rescaling_images(self):
        layer = super().rescaling_images(self.rescaling)
        return layer

    def augmenting_images(self):
        layer = super().augmenting_images(self.random_crop, self.random_flip, self.random_translation, self.random_rotation, self.random_zoom)
        return layer

@hydra.main(config_path="../../configs/", config_name="config")
def main_images(cfg):  
    preprocess = PreprocessingClass_Images(cfg)
    #1. load images / get dataset
    images_train, images_val, images_test = preprocess.load_images()
    print("Dataset_type:", images_train)

    #2. rescaling (rescaling is used in model.py)
    rescaling = preprocess.rescaling_images()
    
    #output as test
    for images, labels in images_train.take(1): #(.take(1) for first batch)
        preprocess.test_output(rescaling(images[0])) #(1. pic of 1. batch)
        print("rescaling done, below image properties:")
        print("image shape:", rescaling(images[0]).shape)
        print("image value range:", rescaling(images[0]).numpy().min(), rescaling(images[0]).numpy().max())

    #3. data augmentation (augmenation is used in model.py)
    augmentation = preprocess.augmenting_images()
    #output as test
    for images, labels in images_train.take(1): #(.take(1) for first batch)
        preprocess.test_output(augmentation(images[0])) #(1. Pic of 1. batch)
        print("augmentation done, below image properties:")
        print("image shape:", augmentation(images[0]).shape)
        print("image value range:", augmentation(images[0]).numpy().min(), augmentation(images[0]).numpy().max())    


if __name__ == "__main__":
    main_timeseries() #tested and worked
    #main_structureddata() #-> tested and worked
    #main_images() #-> tested and worked





### Tasks for Prefect UI ###
#-> tasks has to be outside of any class
#-> class_input equals to self, but not possible due to fact that tasks cannot be in classes
#-> nout=2 cause 2 return values(e.g. images_train/images_val), that causes in output load_data[0]/[1]
#-> checkpoint=False very important by using other frameworks(!), otherwise prefect and tf/keras/sklearn wont work in prefect agent/server

#TimeSeries Tasks
@task(name="timeseries_load_data")
def timeseries_load_data_task(class_input):
    logger = prefect.context.get("logger")
    logger.info("Loading data...")
    return class_input.load_data()

@task(name="timeseries_cleaning_data")
def timeseries_cleaning_data_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("Cleaning data...")
    return class_input.cleaning_data(data)

@task(name="timeseries_preprocessing", checkpoint=False, nout=6)
def timeseries_transform_task(class_input, cleaned_data):
    #split in train/test/val
    x_train, y_train, x_val, y_val, x_test, y_test = class_input.sklearn_split(cleaned_data)
    
    #feature encoding (standardization or normalization or none)
    if class_input.feature_encoding == "standardization":
        x_train = class_input.sklearn_standardization(x_train)
        x_val = class_input.sklearn_standardization(x_val)
        x_test = class_input.sklearn_standardization(x_test)
    elif class_input.feature_encoding == "normalization":
        x_train = class_input.sklearn_normalization(x_train)
        x_val = class_input.sklearn_normalization(x_val)
        x_test = class_input.sklearn_normalization(x_test) 
    
    #label encoding (labelencoding string -> int, onehot or none)
    if class_input.label_encoding == "labelencoding":
        y_train = class_input.sklearn_labelencoder(y_train)
        y_val = class_input.sklearn_labelencoder(y_val)    
        y_test = class_input.sklearn_labelencoder(y_test)   
    elif class_input.label_encoding == "onehot":
        y_train = class_input.sklearn_onehot(y_train)
        y_val = class_input.sklearn_onehot(y_val)
        y_test = class_input.sklearn_onehot(y_test)
    
    #reshaping features (for fiting into 1D CNN)
    x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.values.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train , y_train, x_val, y_val, x_test, y_test


#StructuredData Tasks
@task(name="strucdata_load_data")
def structureddata_load_data_task(class_input):
    logger = prefect.context.get("logger")
    logger.info("Loading data...")
    return class_input.load_data()

@task(name="strucdata_cleaning_data")
def structureddata_cleaning_data_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("Cleaning data...")
    return class_input.cleaning_data(data)

@task(name="sklearn_normalization")
def sklearn_normalization_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("Normalization data...")
    return class_input.sklearn_normalization(data)

@task(name="sklearn_standardization")
def sklearn_standardization_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("Standardization data...")
    return class_input.sklearn_standardization(data)

@task(name="sklearn_onehot")
def sklearn_onehot_task(class_input, data, inverse=False):
    logger = prefect.context.get("logger")
    logger.info("OneHot Encoding data...")
    return class_input.sklearn_onehot(data, inverse)

@task(name="strucdata_tf_transform", checkpoint=False, nout=2) 
def structureddata_tf_transform_task(class_input, dataset):
    logger = prefect.context.get("logger")
    logger.info("Transforming TF Dataset... ")
    all_inputs = []
    encoded_features = []

    #Numeric features (Normalization)
    class_input.tf_set_normalization(class_input.numeric_columns, dataset, all_inputs, encoded_features)
    
    #Categorical features (nominal, onehot encoding)
    #if type=int64 -> integer to onehot
    class_input.tf_set_category_to_onehot(class_input.int_to_onehot_columns, dataset, all_inputs, encoded_features, "int64", 3)
    #if type=string -> string to onehot
    class_input.tf_set_category_to_onehot(class_input.str_to_onehot_columns, dataset, all_inputs, encoded_features, "string", 3) 
    
    #Categorical features (ordinal)
    #string to integer
    class_input.tf_set_category_to_integer(class_input.str_to_int_columns, dataset, all_inputs, encoded_features)

    return all_inputs, encoded_features

@task(name="strucdata_tf_create_dataset", checkpoint=False, nout=3)
def structureddata_tf_dataframe_to_dataset_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("Create Dataset in TF-Format...")
    return class_input.tf_dataframe_to_dataset(data)    


#Images Tasks
@task(name="images_load_data",nout=3,checkpoint=False) 
def images_load_task(class_input):
    logger = prefect.context.get("logger")
    logger.info("Loading data...")
    #logger.info(class_input.load_data()[0].class_names) #logging class_names if needed
    return class_input.load_images()

@task(name="images_rescaling_data",checkpoint=False)
def images_rescaling_task(class_input):
    logger = prefect.context.get("logger")
    logger.info("Creating rescaling layer...")
    return class_input.rescaling_images() 

@task(name="images_augmenting_data",checkpoint=False)
def images_augmenting_task(class_input):
    logger = prefect.context.get("logger")
    logger.info("Creating augmentation layer...")
    return class_input.augmenting_images() 