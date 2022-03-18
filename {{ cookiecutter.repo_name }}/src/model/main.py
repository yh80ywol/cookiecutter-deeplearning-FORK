import sys
sys.path.insert(0,"src/preprocess/")

import preprocess, model
import hydra
from prefect import Flow, Client

#TimeSeries
@hydra.main(config_path="../../configs/", config_name="config")
def timeseries_task(cfg):
    preprocess_class = preprocess.PreprocessingClass_TimeSeries(cfg)
    model_class = model.ModelClass_TimeSeries(cfg)
    with Flow("Train_Model_TimeSeries_Flow") as flow:
        #1. load and preprocess data (for detailed info see preprocessing folder)
        #1.1 load data
        raw_data = preprocess.timeseries_load_data_task(preprocess_class)
        #1.2 clean data
        data = preprocess.timeseries_cleaning_data_task(preprocess_class, raw_data)
        #1.3 transform (with sklearn)
        x_train, y_train, x_val, y_val, x_test, y_test = preprocess.timeseries_transform_task(preprocess_class, data)

        #2. train/validate/test model
        trained_model = model.timeseries_train_model_task(model_class, x_train, y_train, x_val, y_val)
        #3. test model
        model.timeseries_test_model_task(model_class, trained_model, x_test, y_test)

    Client().create_project(project_name=cfg.prefect.project_name)  #create Project if ProjectName not exist, if ProjectName exists nothing happens
    flow.register(project_name=cfg.prefect.project_name) #register flow in ProjectName
    #flow.run() #if want to run immediately 


#StructuredData
@hydra.main(config_path="../../configs/", config_name="config")
def structureddata_task(cfg):
    preprocess_class = preprocess.PreprocessingClass_StructuredData(cfg)
    model_class = model.ModelClass_StructuredData(cfg)
    with Flow("Train_Model_StructuredData_Flow") as flow:
        #1. load and preprocess data (for detailed info see preprocessing folder)
        #1.1 load data
        raw_data = preprocess.structureddata_load_data_task(preprocess_class)
        #1.2 clean data
        data = preprocess.structureddata_cleaning_data_task(preprocess_class, raw_data)
        #1.3 tensorflow/keras
        train_ds, val_ds, test_ds = preprocess.structureddata_tf_dataframe_to_dataset_task(preprocess_class, data)
        #1.4 preprocess tf dataset
        all_inputs, encoded_features = preprocess.structureddata_tf_transform_task(preprocess_class, train_ds)

        #2. train model
        trained_model = model.structureddata_train_model_task(model_class, train_ds, val_ds, all_inputs, encoded_features)
        #3. test model
        model.structureddata_test_model_task(model_class, trained_model, test_ds)

    Client().create_project(project_name=cfg.prefect.project_name)  #create Project if ProjectName not exist, if ProjectName exists nothing happens
    flow.register(project_name=cfg.prefect.project_name) #register flow in ProjectName
    #flow.run() #if want to run immediately 


#Images 
@hydra.main(config_path="../../configs/", config_name="config")
def images_task(cfg):
    preprocess_class = preprocess.PreprocessingClass_Images(cfg)
    model_class = model.ModelClass_Images(cfg)
    with Flow("Train_Model_Image_Flow") as flow:
        #1. load and preprocess data (for detailed info see preprocessing folder)
        #1.1 load images / get dataset
        train_ds, val_ds, test_ds = preprocess.images_load_task(preprocess_class)
        #1.2 rescaling
        rescaling = preprocess.images_rescaling_task(preprocess_class)
        #1.3 data augmentation (RandomCrop, RandomFlip, RandomTransformation, RandomRotation, RandomZoom)
        augmentation = preprocess.images_augmenting_task(preprocess_class)

        #2. train model
        trained_model = model.images_train_model_task(model_class, train_ds, val_ds, augmentation, rescaling)
        #3. test model 
        model.images_test_model_task(model_class, trained_model, test_ds)

    Client().create_project(project_name=cfg.prefect.project_name)  #create Project if ProjectName not exist, if ProjectName exists nothing happens
    flow.register(project_name=cfg.prefect.project_name) #register flow in ProjectName
    #flow.run() #if want to run immediately 


if __name__ == "__main__":
    timeseries_task() #-> tested and worked
    #structureddata_task() #-> tested and worked
    #images_task() #-> tested and worked
