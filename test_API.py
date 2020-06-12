import requests
import numpy as np
import pandas as pd 

text = input("Put an answer...\n")
data = {'value': text}
response = requests.post("{}/".format("http://127.0.0.1:5000"), json =data )
print(str(response.json()))