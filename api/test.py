import requests

r = requests.post("http://localhost:5000/predict", json={
    "displacement": 200,
    "horsepower": 100,
    "weight": 3000,
    "cylinders": 4,
    "origin": 1
})

print("test", r.json())