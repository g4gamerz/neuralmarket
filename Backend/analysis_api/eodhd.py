import requests
import json
url = f'https://eodhd.com/api/exchange-symbol-list/KAR?api_token=6839ee1a3b7e51.24083684&fmt=json'
data = requests.get(url).json()
file_name = 'names.json'
with open(file_name, 'w') as f:
    json.dump(data, f, indent=4)  

print(f"Successfully exported data to {file_name}")