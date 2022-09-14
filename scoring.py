from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

#################Function for model scoring
def score_model(test_data_path = os.path.join(config['test_data_path']), model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")):
    
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    print("Score is calculated for model located at: {0}".format(model_path))
    
    for file in os.listdir(test_data_path):
        if file.endswith(".csv"):
            testdata_path = os.path.join(test_data_path, file)
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    testdata = pd.read_csv(testdata_path)
    
    X_test = testdata.loc[:,["lastmonth_activity", "lastyear_activity","number_of_employees"]].values
    y_test = testdata["exited"].values.reshape(-1,1).ravel()
    
    prediction = model.predict(X_test)
    
    f1 = metrics.f1_score(prediction, y_test)
    
    f1_score_filepath = os.path.join(config['output_model_path'], "latestscore.txt")
    with open(f1_score_filepath, 'w') as fp:
        fp.write("{0}".format(f1))
    
    print("Latest calculated f1 score [{0}] written to: {1}".format(f1, f1_score_filepath))
    return f1

if __name__ == '__main__':
    score_model()