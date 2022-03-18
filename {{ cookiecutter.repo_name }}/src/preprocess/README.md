# Preprocessing


#### start preprocessing timeseries

1. install all requirements 
2. configuration of params in config.yaml (file path, file columns, ...)
3. run: (to run local without flow) 
```
python src/preprocess/preprocess.py
```
-> by default config it will run on TestData "Einpressen_Gesamt.xlsx", sklearn preprocessing and show you relevant transformations on output line, if you want to change data/preprocessings set config.yaml.

5. run: (to register/run flow on prefect, prefect server and UI must be running while executing(!))
```
python src/preprocess/main.py
```



#### Start preprocessing structured data

1. install all requirements 
2. configuration of params in config.yaml (file path, file columns, ...)
3. change comments in preprocess.py and main.py (comment in main_structureddata() in __main__ )
4. run: (to run local without flow) 
```
python src/preprocess/preprocess.py
```
-> by default config it will run on TestData "Einpressen_Gesamt_Test.xlsx", Tensorflow/Keras Preprocessing and show you relevant transformations on output line, if you want to change data/preprocessings set config.yaml.

5. run: (to register/run flow on prefect, prefect server and UI must be running while executing(!))
```
python src/preprocess/main.py
```



#### start preprocessing images

1. install all requirements
2. configuration of params in config.yaml (file path, image size, ...) 
3. change comments in preprocess.py and main.py (comment in main_images() in __main__ )
4. run: (to run local without flow) 
```
python src/preprocess/preprocess.py
```
-> by default config it will run on Images "PEDS1" (images of laser welding process), shows you relevant transformation image data in output line and will display you one transformed image after rescaling and augmentation, if you want to change data/preprocessings set config.yaml.

5. run: (to register/run flow on prefect, prefect server and UI must be running while executing(!))
```
python src/preprocess/main.py
```