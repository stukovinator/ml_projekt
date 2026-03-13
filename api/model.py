from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
import json
import os

print("Pobieranie datasetu...")
dataset = fetch_openml(name='autoMpg', version=1, as_frame=True)
df = dataset.frame.dropna()

X = df.drop(columns=['class'])
y = df['class']

X = pd.get_dummies(X, columns=['cylinders', 'model', 'origin'])
feature_columns = list(X.columns)

print("Trenowanie modelu...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} MPG")
print(f"R²:  {r2:.3f}")

fi = pd.Series(model.feature_importances_, index=feature_columns)
print("\nTop 5 cech:")
print(fi.nlargest(5))

# Zapisz model i metadane
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')

metadata = {
    'feature_columns': feature_columns,
    'mae': round(mae, 3),
    'r2': round(r2, 3),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'feature_importance': {
        col: round(float(val), 4)
        for col, val in fi.nlargest(5).items()
    },
    'mpg_stats': {
        'mean': round(float(y.mean()), 2),
        'min':  round(float(y.min()), 2),
        'max':  round(float(y.max()), 2),
    }
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nZapisano model.pkl i metadata.json — gotowe!")