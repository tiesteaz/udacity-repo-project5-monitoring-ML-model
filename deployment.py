from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

####################function for deployment
def store_model_into_pickle():

    ingested_data_path = os.path.join(config['output_folder_path'])
    output_model_path = os.path.join(config['output_model_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])

    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    ingestedfileslist_filepath = os.path.join(ingested_data_path, "ingestedfiles.txt")
    latestscore_filepath = os.path.join(output_model_path, "latestscore.txt")
    trainedmodel_filepath = os.path.join(output_model_path, "trainedmodel.pkl")
    
    print("Starting production deployment...")

    os.system("cp " + ingestedfileslist_filepath + " " + prod_deployment_path + "/ingestedfiles.txt")
    print("Deployed {0}".format(prod_deployment_path + "/ingestedfiles.txt"))

    os.system("cp " + latestscore_filepath + " " + prod_deployment_path + "/latestscore.txt")
    print("Deployed {0}".format(prod_deployment_path + "/latestscore.txt"))

    os.system("cp " + trainedmodel_filepath + " " + prod_deployment_path + "/trainedmodel.pkl")
    print("Deployed {0}".format(prod_deployment_path + "/trainedmodel.pkl"))

if __name__ == '__main__':
    store_model_into_pickle()