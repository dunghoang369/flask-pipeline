import requests

resp = requests.post("http://localhost:5003/predict", files={"file": open('./cat.jpg','rb')})

print(resp.json())