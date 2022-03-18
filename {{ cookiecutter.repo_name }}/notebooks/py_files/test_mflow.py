import json
from BaseClasses import TrainingBaseClass

#config
with open('config.json', 'r') as f: #Pfad noch "../../configs/config.json" (wenn Template steht)
    config = json.load(f)

class Training(TrainingBaseClass):
    def __init__(self, config):
        self.confing = config


Test=Training(config)
Training.train(1)

#Tracking UI
#run comand "mlflow ui" in CMD bei Path wo Datei liegt!

#Error:
#Could not find experiment with ID 1
#-> unter Tracking UI auf Experiment erstellen (Experiment ID wird dann angezeigt)

