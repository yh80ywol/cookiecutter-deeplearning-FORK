#TODO LISTE
#-> wenn genaue Struktur der images in den json files vorhanden,
#-> images noch testen bzw. base64 strings noch in "echte" images umwandeln (bisher als json file abgespeichert)
    
from dataloader_baseclass import *
import prefect, hydra, requests, json
from prefect import task

class DataloaderID(DataloaderBaseClass):
    def __init__(self, config):
        self.url_root = config.dataloader.RestEndpoint
        self.file_type = config.dataloader.filetype_ID
        self.path = config.dataloader.path_ID
        self.file_name = config.dataloader.filename_ID
        self.output_data = list()

        self.start = config.dataloader.startdate
        self.end = config.dataloader.enddate
        self.process_name = config.dataloader.process_name
        self.operator_first_name = config.dataloader.operator_first_name
        self.operator_family_name = config.dataloader.operator_family_name
        self.material_type = config.dataloader.material_type

    def execute_request(self):
        #build request_url
        #e.g. :
            #Endpoint: http://131.188.113.42:8050/factsfortime
            #Parameter: ?datestart=2021-08-10
            #Requesturl: http://131.188.113.42:8050/factsfortime?datestart=2021-08-10
            #-> so all data from datestart until now will be requested from database

        self.request_url = self.url_root 
        if self.start: #check if string is empty or not
            self.request_url += "?datestart=" + self.start
        if self.end:
            self.request_url += "&dateend=" + self.end
        if self.process_name:
            self.request_url += "&process_name=" + self.process_name
        if self.operator_first_name:
            self.request_url += "&operator_first_name=" + self.operator_first_name
        if self.operator_family_name:
            self.request_url += "&operator_family_name=" + self.operator_family_name 
        if self.material_type:
            self.request_url += "&material_type=" + self.material_type             

        self.request_call = requests.get(self.request_url, verify = False)
        self.request_data = self.request_call.json()
        print("Request data per API from URL = ", self.request_url)
        return self.request_data

    def parse_data(self, data):
        self.output_data.extend(data) 
        print("Parsing Facts...")
        return self.output_data
        
    def save_data(self, data):
        super().save_data(self.path, self.file_name, self.file_type, data)
        print("Saving Facts...")


class DataloaderMeasurements(DataloaderBaseClass):

    def __init__(self, config):
        self.file_type = config.dataloader.filetype_measurements
        self.path = config.dataloader.path_measurements
        self.file_name = config.dataloader.filename_measurements
        self.output_data = list()

    def parse_data(self, data):
        for f in data:
            value = dict()
            value["id"] = f.get("id")
            value["measurements"] = f.get("measurements")
            self.output_data.append(value)
        print("Parsing Measurements...")
        return self.output_data

    def save_data(self, data):
        super().save_data(self.path, self.file_name, self.file_type, data)
        print("Saving Measurements...")


class DataloaderImages(DataloaderBaseClass):
    
    def __init__(self, config):
        self.file_type = config.dataloader.filetype_img
        self.path = config.dataloader.path_img
        self.file_name = config.dataloader.filename_img
        self.output_data = list()
        
    def parse_data(self, data):
        for f in data:
            value = dict()
            value["data_files"] = f.get("data_files")
            self.output_data.append(value)
        print("Parsing images...")
        return self.output_data

    def save_data(self, data):
        super().save_data(self.path, self.file_name, self.file_type, data)
        print("Saving images...")


@hydra.main(config_path="../../configs/", config_name="config")
def main(cfg):
    #Die Daten werden mithilfe einer GET-Methode und entsprechenden Parametern, welche in der Konfigurationsdatei eingestellt werden können, abgerufen:
#Endpoint: http://131.188.113.42:8050/factsfortime
#	Parameter: ?datestart=2021-08-10
##	Request: http://131.188.113.42:8050/factsfortime?datestart=2021-08-10
#Mit dem oben gezeigten Request werden nun beispielsweise alle Daten geladen, die nach dem 10.08.2021 auf der Datenbank hinzugefügt wurden. Weitere Parameter können in der Konfigurationsdatei nachgeschlagen werden. 

    #1. facts
    facts = DataloaderID(cfg)
    facts.execute_request() #execute request

    #1.1 get all facts (including ID, Measurements, Images, etc...)
    facts.__get__(facts.request_data) #get data
    facts.save_data(facts.output_data) #save data

    #1.2 get data per iterative loop, e.g. 0-10 data & dont exceed max. length dataset)
        #nums = facts.__next__(0, 10, facts.request_data)
        #facts.parse_data(next(nums)) #first entry
        #facts.parse_data(next(nums)) #second entry, ...
        #facts.save_data(facts.output_data) #save entrys
        #for num in nums: #loop über alle entrys
        #    facts.parse_data(num)
        #facts.save_data(facts.output_data) #save entrys
    print("Facts loaded successfully !")


    #2. measurements
    measurements = DataloaderMeasurements(cfg)

    #2.1 get all measurements
    measurements.__get__(facts.output_data) #get measurements
    measurements.save_data(measurements.output_data) #save measurements

    #2.2 get data per iterative loop, e.g. 0-10 data & dont exceed max. length dataset)
        #nums = measurements.__next__(0, 10, facts.output_data) #next
        #measurements.parse_data(next(nums)) #first entry
        #measurements.parse_data(next(nums)) #second entry, ...
        #measurements.save_data(measurements.output_data) #save
    print("Measurements loaded successfully !")


    #3. images
    images = DataloaderImages(cfg)

    #3.1 get all images
    images.__get__(facts.output_data) #get images
    images.save_data(images.output_data) #save images

    #3.2 get data per iterative loop, e.g. 0-10 data & dont exceed max. length dataset)
        #nums = images.__next__(0, 10, facts.output_data) #next
        #images.parse_data(next(nums)) #first entry
        #images.parse_data(next(nums)) #second entry, ...
        #images.save_data(images.output_data) #save
    print("Images loaded successfully !")


if __name__ == "__main__":
    main() #-> tested and worked

    


### Tasks for Prefect UI ###
#-> tasks has to be outside of any class
#-> class_input equals to self, but not possible due to fact that tasks cannot be in classes

#Facts
@task(name="execute_request")
def execute_request_task(class_input):
    logger = prefect.context.get("logger")
    logger.info("execute request...")
    return class_input.execute_request()
@task(name="save_data")
def facts_save_data_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("save data...")
    return class_input.save_data(data)

#Measurements
@task(name="measurements_parse_data")
def measurements_parse_data_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("parse data...")
    return class_input.parse_data(data)
@task(name="measurements_save_data")
def measurements_save_data_task(class_input, measurements):
    logger = prefect.context.get("logger")
    logger.info("save data...")
    return class_input.save_data(measurements)

#Images
@task(name="images_parse_data")
def images_parse_data_task(class_input, data):
    logger = prefect.context.get("logger")
    logger.info("parse data...")
    return class_input.parse_data(data)
@task(name="images_save_data")
def images_save_data_task(class_input, images):
    logger = prefect.context.get("logger")
    logger.info("save data...")
    return class_input.save_data(images)
