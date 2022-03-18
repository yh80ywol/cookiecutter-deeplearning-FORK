# Model Training


#### start model training timeseries

1. install all requirements 
2. configuration of params in config.yaml (learning rate, callbacks, ...)
3. run: (to run local without flow) 
```
python src/model/model.py
```
-> by default config it will run on TestDataset "Einpressen_Gesamt" and 1DCNN. After Training new model and images of training/layout saved saved in folder by default in reports/mlruns, if you want to change datasets/configurations check model.py and config.yaml.

4. run: (to register/run flow on prefect, prefect server and UI must be running while executing(!))
```
python src/model/main.py
```



#### start model training structured data

1. install all requirements 
2. configuration of params in config.yaml (learning rate, callbacks, ...)
3. change comments in model.py and main.py (comment in main_structureddata() in __main__ )
4. run: (to run local without flow) 
```
python src/model/model.py
```
-> by default config it will run on TestDataset "Heart Disease" and simple neural network. After Training new model and images of training/layout saved saved in folder by default in reports/mlruns, if you want to change datasets/configurations check model.py and config.yaml.

5. run: (to register/run flow on prefect, prefect server and UI must be running while executing(!))
```
python src/model/main.py
```



#### start model training images

1. install all requirements
2. configuration of params in config.yaml (learning rate, callbacks, ...)
3. change comments in model.py and main.py (comment in main_images() in __main__ )
4. run: (to run local without flow) 
```
python src/model/model.py
```
-> by default config it will run on Images "PEDS1" (450 images of laser welding process) and ResNet50 basemodel. After Training new model and images of training/layout saved saved in folder by default in reports/mlruns, if you want to change datasets/configurations check model.py and config.yaml.

5. run: (to register/run flow on prefect, prefect server and UI must be running while executing(!))
```
python src/model/main.py
```