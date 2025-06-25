import requests

url = f'https://eodhd.com/api/real-time/HBL.KAR?api_token=6839ee1a3b7e51.24083684&fmt=json'
data = requests.get(url).json()

print(data)