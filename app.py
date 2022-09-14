from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis 
import diagnostics
import scoring
#import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','POST','OPTIONS'])
def predict():
    
    testdata_filepath = request.args.get('testdata_filepath')

    #call the prediction function you created in Step 3
    
    if testdata_filepath != None:
        predictions = diagnostics.model_predictions(pd.read_csv(testdata_filepath))
    else:
        predictions = []
    
    return str(predictions) + "\n", 200 #add return value for prediction outputs


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    
    score = scoring.score_model()
    return str(score) + "\n", 200 #add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    #check means, medians, and modes for each column
    summary = diagnostics.dataframe_summary()
    
    return str(summary) + "\n", 200 #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_stats():        
    #check timing and percent NA values
    
    d_stats = "% of NaN in the test data columns: " + str(diagnostics.missing_data()) + "\n"
    d_stats += "Execution time, sec [ingestion, training]: " + str(diagnostics.execution_time()) + "\n"
    d_stats += "Outdated packages found: " + str(diagnostics.outdated_packages_list())
    return d_stats + "\n", 200 #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
