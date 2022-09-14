import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

##############Function for reporting
def score_model():
    
    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])
    outputmodel_path = os.path.join(config['output_model_path'], "confusionmatrix.png")
    
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    for file in os.listdir(test_data_path):
        if file.endswith(".csv"):
            testdata_path = os.path.join(test_data_path, file)
    
    print("Using test data from: {0}".format(testdata_path))
    
    testdata = pd.read_csv(testdata_path)
    
    # obtain test data from test dataset
    y_test = testdata["exited"].values.reshape(-1,1).ravel()

    # calculate predictions
    y_predicted = diagnostics.model_predictions(testdata)
    
    print('y_test: {0}'.format(y_test))
    print('y_predicted: {0}'.format(y_predicted))
    
    plt.clf()

    conf_matrix = confusion_matrix(y_test, y_predicted)
    
    print("Confusion matrix:")
    print(conf_matrix)
    
    # copied idea from https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib
    _, ax = plt.subplots()
    ax.matshow(conf_matrix)

    for (i, j), z in np.ndenumerate(conf_matrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    plt.savefig(outputmodel_path)
    
    return

if __name__ == '__main__':
    score_model()
