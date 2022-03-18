import sys
sys.path.insert(0,"src/dataloader/")
sys.path.insert(0,"src/preprocess/")
sys.path.insert(0,"src/model/")

##Preprocess Flow
from src.dataloader.main import dataloader_task as dataloader

#Preprocess Flows (if you just want to register preprocess flows)
from src.preprocess.main import timeseries_task as preprocess_timeseries
from src.preprocess.main import images_task as preprocess_images
from src.preprocess.main import structureddata_task as preprocess_strucdata

#Training Model Flows (model flows include preprocess flows)
from src.model.main import timeseries_task as train_timeseries
from src.model.main import images_task as train_images
from src.model.main import structureddata_task as train_strucdata

if __name__ == "__main__":
    #dataloader() #tested and worked (NEED connection to database, e.g via cisco)

    train_timeseries() #tested and worked
    train_images() #tested and worked
    train_strucdata() #tested and worked