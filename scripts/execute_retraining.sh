#!/bin/bash

# Retrain the model
python prepare_to_retrain.py
python download_data.py
python link_inspections_violations.py
python prepare_nei_data.py
python build_evaluate_model.py
python tests.py

# Check for number of assertion errors
num_AEs=$(tr ' ' '\n' < logfile.txt | grep AssertionError | wc -l)

# If none, commit changes to web-app and upload to heroku
if [ $num_AEs -eq 0 ]
then 
    cd ./web-app
    git add .
    git commit -m "Model retrained: $(date)"
    #git push epa-air-violations master
fi

