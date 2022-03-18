import preprocess
import hydra
from prefect import Flow, Client

@hydra.main(config_path="../../configs/", config_name="config")
def timeseries_task(cfg):
    preprocess_class = preprocess.PreprocessingClass_TimeSeries(cfg)
    with Flow("Preprocess_TimeSeries_Flow") as flow:
        #1. load data
        raw_data = preprocess.timeseries_load_data_task(preprocess_class)
        #2. clean data
        data = preprocess.timeseries_cleaning_data_task(preprocess_class, raw_data)
        #3. transform (with sklearn)
        x_train, y_train, x_val, y_val, x_test, y_test = preprocess.timeseries_transform_task(preprocess_class, data)
        
    Client().create_project(project_name=cfg.prefect.project_name)  #create Project if ProjectName not exist, if ProjectName exists nothing happens
    flow.register(project_name=cfg.prefect.project_name) #register flow in ProjectName
    #flow.run() #if want to run immediately


@hydra.main(config_path="../../configs/", config_name="config")
def structureddata_task(cfg):
    preprocess_class = preprocess.PreprocessingClass_StructuredData(cfg)
    with Flow("Preprocess_StructuredData_Flow") as flow:
        #1. load data
        raw_data = preprocess.structureddata_load_data_task(preprocess_class)
        #2. clean data
        data = preprocess.structureddata_cleaning_data_task(preprocess_class,raw_data)
        #3. sklearn
            #normalization (d.h. values in range 0-1)
            #normalized_data = preprocess.sklearn_normalization_task(preprocess_class, data)
            #standardization (here no range 0-1(!), but mean=0 and std=1, can cause problems by NN, but better for statistic outliner)
            #standardized_data = preprocess.sklearn_standardization_task(preprocess_class, data)
            #onehot encoding (ordinal cat. var. e.g. t-shirt size S<M<L, or nominal cat. var. e.g. colors of t-shirt(here no "ranking"))
            #onehot_data = preprocess.sklearn_onehot_task(preprocess_class, data, inverse=False)
            #onehot_data_inverse = preprocess.sklearn_onehot_task(preprocess_class, data, inverse=True)
        #4. tensorflow/keras
        train_ds, val_ds, test_ds = preprocess.structureddata_tf_dataframe_to_dataset_task(preprocess_class,data)
        #4.1 preprocess tf dataset
        all_inputs, encoded_features = preprocess.structureddata_tf_transform_task(preprocess_class,train_ds)

    Client().create_project(project_name=cfg.prefect.project_name)  #create Project if ProjectName not exist, if ProjectName exists nothing happens
    flow.register(project_name=cfg.prefect.project_name) #register flow in ProjectName
    #flow.run() #if want to run immediately


@hydra.main(config_path="../../configs/", config_name="config")
def images_task(cfg):
    preprocess_class = preprocess.PreprocessingClass_Images(cfg)
    with Flow("Preprocess_Image_Flow") as flow:
        #1. load images / get dataset
        images_train, images_val, images_test = preprocess.images_load_task(preprocess_class)
        #2. rescaling
        rescaling = preprocess.images_rescaling_task(preprocess_class)
        #3. data augmentation (RandomCrop, RandomFlip, RandomTransformation, RandomRotation, RandomZoom)
        augmentation = preprocess.images_augmenting_task(preprocess_class)
    
    Client().create_project(project_name=cfg.prefect.project_name)  #create Project if ProjectName not exist, if ProjectName exists nothing happens
    flow.register(project_name=cfg.prefect.project_name) #register flow in ProjectName
    #flow.run() #if want to run immediately


if __name__ == "__main__":
    timeseries_task() #-> tested and worked
    #structureddata_task() #-> tested and worked
    #images_task() #-> tested and worked
