import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

numeric_columns = ['lastmonth_activity','lastyear_activity','number_of_employees']    

    
##################Function to get model predictions
def model_predictions(testdata):

    model_path = os.path.join(config['prod_deployment_path'], "trainedmodel.pkl")

    try:
        #read the deployed model and a test dataset, calculate predictions
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        X_test = testdata.loc[:,numeric_columns].values

        predictions = model.predict(X_test)
    except:
        predictions = []
        
    return predictions # return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 

    # read datafile into pandas dataframe
    data = pd.read_csv(dataset_csv_path)
    
    # calculate means of numeric columns
    
    means = data[numeric_columns].mean(numeric_only = True)
    print("Means {0}: ".format(numeric_columns))
    print(means.to_list())
    print('')
    
    # calculate medians of numeric columns
    medians = data[numeric_columns].median(numeric_only = True)
    print("Medians {0}: ".format(numeric_columns))
    print(medians.to_list())
    print('')
    
    # calculate standard deviations of numeric columns
    stds = data[numeric_columns].std(numeric_only = True)
    print("Standard deviations {0}: ".format(numeric_columns))
    print(stds.to_list())
    print('')
    
    return [means.to_list(), medians.to_list(), stds.to_list()] #return value should be a list containing all summary statistics

##################Function to get timings
def missing_data():
    dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 

    #calculate percentage of missing values in data columns
    data = pd.read_csv(dataset_csv_path)
    
    countNaNinData = list(data[numeric_columns].isna().sum())
    pctOfNaNValuesInData = [countNaNinData[i]/len(data.index) for i in range(len(countNaNinData))]
    print("Pct of rows with NaN in data columns: {0}".format(pctOfNaNValuesInData))
    print("")
    
    return pctOfNaNValuesInData #return a list of percentage of NaN  in 'data' columns

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    
    ingestion_start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing = timeit.default_timer() - ingestion_start_time
    print("Ingestion timing, sec: {0}".format(ingestion_timing))
    print("")
    
    training_start_time = timeit.default_timer()
    os.system('python3 training.py')
    training_timing = timeit.default_timer() - training_start_time
    print("Training timing, sec: {0}".format(training_timing))
    print("")
    
    timings = [] # list for training time, and ingestion time
    
    timings.append(ingestion_timing)
    timings.append(training_timing)
        
    return timings #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    
    outdated_packages = subprocess.run(["python", "-m", "pip", "list", "--outdated"], capture_output = True).stdout
    outdated_packages = outdated_packages.decode("utf-8")
    return outdated_packages

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
