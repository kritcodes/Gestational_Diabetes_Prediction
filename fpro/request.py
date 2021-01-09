import requests

url = 'http://localhost:5000/predict'
r = requests.post(url,json={'preg':5, 'glucose':110, 'bloodpres':68, 'skinthick:'0, 'insulin:'0,'bmi:'26, 'dpf:'0.292})

print(r.json())