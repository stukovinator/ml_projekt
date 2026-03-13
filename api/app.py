from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("model/model.pkl")
with open("model/metadata.json") as f:
    metadata = json.load(f)

@app.route("/stats")
def stats():
    return jsonify(metadata)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    row = {col: 0 for col in metadata["feature_columns"]}

    row['displacement'] = float(data.get('displacement', 200))
    row['horsepower']   = float(data.get('horsepower', 100))
    row['weight']       = float(data.get('weight', 3000))
    row['acceleration'] = float(data.get('acceleration', 15))
    model_col = f"model_{int(data.get('model_year', 76))}"
    if model_col in row:
        row[model_col] = 1

    cyl_col = f"cylinders_{data.get("cylinders", 4)}"
    if cyl_col in row:
        row[cyl_col] = 1
    
    origin_col = f"origin_{data.get('origin', 1)}"
    if origin_col in row:
        row[origin_col] = 1

    X = pd.DataFrame([row])
    mpg = float(model.predict(X)[0])

    l_per_100km = round(235.214 / mpg, 2)

    return jsonify({
        "mpg": round(mpg, 1),
        "l_per_100km": l_per_100km,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)

