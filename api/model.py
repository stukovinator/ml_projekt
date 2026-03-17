import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

dfs = []
COL_MAP_OLD = {
    'ENGINE SIZE': 'engine_size', 'CYLINDERS': 'cylinders',
    'FUEL': 'fuel_type', 'VEHICLE CLASS': 'vehicle_class',
    'Unnamed: 10': 'combined_l100', 'CO2 EMISSIONS ': 'co2',
}
COL_MAP_NEW = {
    'Engine size (L)': 'engine_size', 'Cylinders': 'cylinders',
    'Fuel type': 'fuel_type', 'Vehicle class': 'vehicle_class',
    'Combined (L/100 km)': 'combined_l100', 'CO2 emissions (g/km)': 'co2',
}

for year in [2015, 2021, 2022, 2023, 2024]:
    url = f"https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY{year}%20Fuel%20Consumption%20Ratings.csv"
    df = pd.read_csv(url, encoding='latin-1', low_memory=False)
    if 'FUEL CONSUMPTION*' in df.columns or 'FUEL CONSUMPTION' in df.columns:
        df = df.rename(columns=COL_MAP_OLD).iloc[1:].reset_index(drop=True)
    else:
        df = df.rename(columns=COL_MAP_NEW)
    needed = ['engine_size','cylinders','fuel_type','vehicle_class','combined_l100','co2']
    if all(c in df.columns for c in needed):
        df = df[needed].copy()
        df['year'] = year
        dfs.append(df)
        print(f"{year}: {len(df)} rekordów")

df = pd.concat(dfs, ignore_index=True)

for col in ['combined_l100','engine_size','cylinders','co2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['combined_l100','engine_size','cylinders'])
df = df[df['fuel_type'].isin(['X','Z','D'])]
df = df[df['combined_l100'].between(3, 25)]

print(f"\nŁącznie: {df.shape}")
print(df['fuel_type'].value_counts())
print(df['combined_l100'].describe())

le_class = LabelEncoder()
le_fuel  = LabelEncoder()
df['vehicle_class_enc'] = le_class.fit_transform(df['vehicle_class'].astype(str))
df['fuel_type_enc']     = le_fuel.fit_transform(df['fuel_type'].astype(str))

features = features = ['engine_size', 'cylinders', 'vehicle_class_enc', 'fuel_type_enc']
X = df[features].dropna()
y = df.loc[X.index, 'combined_l100']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2  = r2_score(y_test, model.predict(X_test))
print(f"\nMAE: {mae:.2f} L/100km")
print(f"R²:  {r2:.3f}")

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')

fi = pd.Series(model.feature_importances_, index=features)
metadata = {
    'features': features,
    'mae': round(mae, 3),
    'r2':  round(r2, 3),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'feature_importance': {k: round(float(v), 4) for k,v in fi.nlargest(5).items()},
    'vehicle_classes': list(le_class.classes_),
    'fuel_types': list(le_fuel.classes_),
}
with open('model/metadata.json','w') as f:
    json.dump(metadata, f, indent=2)

joblib.dump({'class': le_class, 'fuel': le_fuel}, 'model/encoders.pkl')
print("\nZapisano model.pkl, metadata.json, encoders.pkl — gotowe!")