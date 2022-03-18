# Dataloader


#### Start loading data

1. install all requirements 
2. configuration of params in config.yaml (file path, startdate, enddate, ...)
3. run: (to run local without flow)
```
python src/dataloader/dataloader.py
```

4. run: (to register/run flow on prefect, prefect server and UI must be running while executing(!))
```
python src/dataloader/main.py
```

