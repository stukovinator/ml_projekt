from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import pandas as pd

app = Flask(__name__)
CORS(app)

model    = joblib.load('model/model.pkl')
encoders = joblib.load('model/encoders.pkl')
with open('model/metadata.json') as f:
    metadata = json.load(f)

@app.route('/stats')
def stats():
    return jsonify(metadata)

@app.route('/vehicle_classes')
def vehicle_classes():
    return jsonify(metadata['vehicle_classes'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    vehicle_class = data.get('vehicle_class', 'Compact')
    fuel_type     = data.get('fuel_type', 'X')

    # Enkoduj tak samo jak przy trenowaniu
    try:
        vc_enc = encoders['class'].transform([vehicle_class])[0]
    except ValueError:
        vc_enc = 0

    try:
        ft_enc = encoders['fuel'].transform([fuel_type])[0]
    except ValueError:
        ft_enc = 0

    row = {
        'engine_size':       float(data.get('engine_size', 2.0)),
        'cylinders':         int(data.get('cylinders', 4)),
        'vehicle_class_enc': vc_enc,
        'fuel_type_enc':     ft_enc,
    }

    X = pd.DataFrame([row])
    l100 = float(model.predict(X)[0])
    l100 = round(l100, 1)

    dist       = float(data.get('distance', 500))
    total      = round((l100 / 100) * dist, 1)
    jerry_cans = -(-int(total) // 20)  # ceiling division

    return jsonify({
        'l_per_100km': l100,
        'total_liters': total,
        'jerry_cans': jerry_cans,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)