import dataloader
from prefect import Flow, Client
import hydra

@hydra.main(config_path="../../configs/", config_name="config")
def main(cfg):
    facts = dataloader.DataloaderID(cfg) #facts
    measurements = dataloader.DataloaderMeasurements(cfg) #measurements
    images = dataloader.DataloaderImages(cfg) #images
    
    with Flow("Dataloader_Flow") as flow:
        #extract data from API
        request_data = dataloader.execute_request_task(facts)
        
        #parse data (measurements/images)
        measurements_data = dataloader.measurements_parse_data_task(measurements, request_data)
        images_data = dataloader.images_parse_data_task(images, request_data)

        #save data (facts/measurements/images)
        dataloader.facts_save_data_task(facts, request_data)
        dataloader.measurements_save_data_task(measurements, measurements_data)
        dataloader.images_save_data_task(images, images_data)

    Client().create_project(project_name=cfg.prefect.project_name)  #create Project if ProjectName not exist, if ProjectName exists nothing happens
    flow.register(project_name=cfg.prefect.project_name) #register flow in ProjectName
    #flow.run() #if want to run immediately 


if __name__ == "__main__":
    main() #-> tested and worked
    
