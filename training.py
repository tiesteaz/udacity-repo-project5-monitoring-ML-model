from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

#################Function for training the model
def train_model():
    dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 
    model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl") 

    trainingdata = pd.read_csv(dataset_csv_path)

    X = trainingdata.loc[:,["lastmonth_activity", "lastyear_activity","number_of_employees"]].values
    y = trainingdata["exited"].values.reshape(-1,1).ravel()

    #print(trainingdata)

    #use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    n_jobs=None, penalty='l2', multi_class='auto', #multi_class='warn',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model = logit.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(model_path, 'wb'))
    print("Trained model written to: {0}".format(model_path))

if __name__ == '__main__':
    train_model()
    