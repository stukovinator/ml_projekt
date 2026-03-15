import requests
r = requests.post("http://localhost:5000/predict", json={
    "engine_size": 1.9,
    "cylinders": 4,
    "vehicle_class": "Compact",
    "fuel_type": "D",
    "distance": 500
})
print(r.json())