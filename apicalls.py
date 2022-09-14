import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

#Call each API endpoint and store the responses
response1 = requests.get(URL + "prediction?testdata_filepath=testdata/testdata.csv").content
response2 = requests.get(URL + "scoring").content
response3 = requests.get(URL + "summarystats").content
response4 = requests.get(URL + "diagnostics").content


#combine all API responses
responses = response1 + response2 + response3 + str(response4)

#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f) 
    
output_path = os.path.join(config['output_model_path'], "apireturns.txt")

with open(output_path, "w") as f:
    f.write(responses)