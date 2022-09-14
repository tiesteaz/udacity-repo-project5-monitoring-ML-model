import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

#############Function for data ingestion
def merge_multiple_dataframe():
    
    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']

    ingestedfiles_filename = "ingestedfiles.txt"
    outputfilename = "finaldata.csv"

    ingestedfiles_filepath = os.path.join(output_folder_path, ingestedfiles_filename)
    outputfilepath = os.path.join(output_folder_path, outputfilename)

    ingestedfiles_list = []

    # initializing final dataframe
    dataframe = pd.DataFrame()
    
    # reading list of files into the data frame
    for inputfile in os.listdir(input_folder_path):
        
        # read ingested file
        inputfilepath = os.path.join(input_folder_path, inputfile)
        
        try:
            df = pd.read_csv(inputfilepath)
            dataframe = dataframe.append(df).reset_index(drop=True)

            # deduplicate final dataframe
            dataframe = dataframe.drop_duplicates().reset_index(drop=True)

            # record ingested file in the list
            ingestedfiles_list.append(inputfile)
        except:
            continue

    # save final deduplicated dataframe to file
    dataframe.to_csv(outputfilepath)
    
    # save the list of ingested files into the txt file
    with open(ingestedfiles_filepath, 'w') as fp:
        fp.write("\n".join(ingestedfiles_list))
    
    print("Ingested output file written to: {0}".format(outputfilepath))
    print("Ingestedfiles.txt written to: {0}".format(ingestedfiles_filepath))

if __name__ == '__main__':
    merge_multiple_dataframe()