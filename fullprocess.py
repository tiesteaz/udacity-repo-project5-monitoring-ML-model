import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

import sys
import os
import json
import pickle
import pandas as pd
from sklearn import metrics

with open('config.json','r') as f:
    config = json.load(f) 

##################Check and read new data
#first, read ingestedfiles.txt from prod_deployment_path
ingestedfiles_path = os.path.join(config['prod_deployment_path'], "ingestedfiles.txt")

with open(ingestedfiles_path, "r") as f:
    ingested_files = [line.rstrip() for line in f]
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
current_files = os.listdir(config['input_folder_path'])

new_files = []

for file in ingested_files:
    new_files.append(file)

for file in current_files:
    new_files.append(file)

new_files = list(dict.fromkeys(new_files)) # deduplicate filenames

for old_file in ingested_files:
    if old_file in new_files:
        new_files.remove(old_file)

if len(new_files) == 0:
    sys.exit("No new files found to ingest. Exiting...")

print("New files found to ingest: {0}".format(new_files))
print("")

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
ingestion.merge_multiple_dataframe()

##################Checking for model drift
# get test data from the latest obtained files
test_data_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['prod_deployment_path'], "trainedmodel.pkl")

# calculate new_f1_score
new_f1_score = scoring.score_model(test_data_path, model_path)

#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(config['prod_deployment_path'], "latestscore.txt")) as f:
    last_f1_score = float(f.read())

print("Last f1 score: {0}\r\n".format(last_f1_score))
print("New f1 score: {0}\r\n".format(new_f1_score))

model_drift = new_f1_score < last_f1_score # raw comparison test

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if not(model_drift):  
    sys.exit("No model drift detected, exiting...")

f1_score_filepath = os.path.join(config['output_model_path'], "latestscore.txt")
with open(f1_score_filepath, 'w') as fp:
    fp.write("{0}".format(new_f1_score))

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
os.system("python training.py")
os.system("python scoring.py")
os.system("python deployment.py")

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system("python reporting.py")
os.system("python2 apicalls.py")