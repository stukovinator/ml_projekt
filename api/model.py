#
#  model.py: trenowanie modelu predykcji zużycia paliwa
#  Dataset: Natural Resources Canada (NRCan), lata 2015–2024
#  Algorytm: Random Forest Regressor
#

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os


# oficjalne dane rządu Kanady — zużycie paliwa dla każdego
# modelu samochodu dostępnego w sprzedaży (L/100km)

print("Pobieranie danych z NRCan...")

KOLUMNY_STARE = {
    'ENGINE SIZE':  'engine_size',
    'CYLINDERS':    'cylinders',
    'FUEL':         'fuel_type',
    'VEHICLE CLASS':'vehicle_class',
    'Unnamed: 10':  'combined_l100',
}
KOLUMNY_NOWE = {
    'Engine size (L)':     'engine_size',
    'Cylinders':           'cylinders',
    'Fuel type':           'fuel_type',
    'Vehicle class':       'vehicle_class',
    'Combined (L/100 km)': 'combined_l100',
}

tabele = []
for rok in [2015, 2021, 2022, 2023, 2024]:
    url = f"https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY{rok}%20Fuel%20Consumption%20Ratings.csv"
    df = pd.read_csv(url, encoding='latin-1', low_memory=False)

    # wykryj format pliku po nazwie kolumny i przemianuj
    if 'FUEL CONSUMPTION*' in df.columns or 'FUEL CONSUMPTION' in df.columns:
        df = df.rename(columns=KOLUMNY_STARE).iloc[1:].reset_index(drop=True)
    else:
        df = df.rename(columns=KOLUMNY_NOWE)

    potrzebne = ['engine_size', 'cylinders', 'fuel_type', 'vehicle_class', 'combined_l100']
    if all(k in df.columns for k in potrzebne):
        tabele.append(df[potrzebne].assign(rok=rok))
        print(f"  {rok}: {len(df)} rekordów")

# połącz wszystkie lata
df = pd.concat(tabele, ignore_index=True)
print(f"Łącznie: {len(df)} rekordów\n")


# zamieniamy tekst na liczby, usuwamy brakujące wartości
# zostawiamy tylko pojazdy benzynowe i diesle

df['combined_l100'] = pd.to_numeric(df['combined_l100'], errors='coerce')
df['engine_size']   = pd.to_numeric(df['engine_size'],   errors='coerce')
df['cylinders']     = pd.to_numeric(df['cylinders'],     errors='coerce')

df = df.dropna(subset=['combined_l100', 'engine_size', 'cylinders'])
df = df[df['fuel_type'].isin(['X', 'Z', 'D'])]       # tylko benzyna i diesel
df = df[df['combined_l100'].between(3, 25)]            # usuń nierealne wartości


# ML nie rozumie tekstu, LabelEncoder zamienia na cyfry

koder_klasy  = LabelEncoder()
koder_paliwa = LabelEncoder()

df['vehicle_class_enc'] = koder_klasy.fit_transform(df['vehicle_class'].astype(str))
df['fuel_type_enc']     = koder_paliwa.fit_transform(df['fuel_type'].astype(str))

# X = cechy wejściowe
# y = zmienna docelowa

CECHY = ['engine_size', 'cylinders', 'vehicle_class_enc', 'fuel_type_enc']

X = df[CECHY]
y = df['combined_l100']

# 80% danych do nauki, 20% do sprawdzenia skuteczności
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Zbiór treningowy: {len(X_train)} próbek")
print(f"Zbiór testowy:    {len(X_test)} próbek\n")


# Random Forest = 100 drzew decyzyjnych
# każde drzewo uczy się na losowym podzbiorze danych
# wynik końcowy to średnia ze wszystkich drzew.

print("Trenowanie modelu...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ocena skuteczności:
# MAE — średni błąd w L/100km
# R²  — jak dobrze model dopasował się do danych

mae = mean_absolute_error(y_test, model.predict(X_test))
r2  = r2_score(y_test,           model.predict(X_test))

print(f"MAE: {mae:.2f} L/100km")
print(f"R²:  {r2:.3f}\n")

# model.pkl     — wytrenowany model
# encoders.pkl  — kodery kategorii (potrzebne przy predykcji)
# metadata.json — statystyki wyświetlane na stronie

os.makedirs('model', exist_ok=True)

joblib.dump(model, 'model/model.pkl')
joblib.dump({'class': koder_klasy, 'fuel': koder_paliwa}, 'model/encoders.pkl')

fi = pd.Series(model.feature_importances_, index=CECHY).sort_values(ascending=False)

metadata = {
    'features':           CECHY,
    'mae':                round(mae, 3),
    'r2':                 round(r2, 3),
    'training_samples':   len(X_train),
    'test_samples':       len(X_test),
    'feature_importance': {k: round(float(v), 4) for k, v in fi.items()},
    'vehicle_classes':    list(koder_klasy.classes_),
    'fuel_types':         list(koder_paliwa.classes_),
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)


raport = f"""================================================================
 SPALO — RAPORT MODELU ML
 Wygenerowano: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}
================================================================

DATASET
  Zrodlo:              Natural Resources Canada (NRCan)
  Lata:                2015, 2021-2024
  Rekordow lacznie:    {len(df)}
  Probek treningowych: {len(X_train)}
  Probek testowych:    {len(X_test)}

ALGORYTM
  Random Forest Regressor
  Liczba drzew:   100
  random_state:   42

CECHY WEJSCIOWE
  {CECHY[0]:<25} pojemnosc silnika w litrach
  {CECHY[1]:<25} liczba cylindrow
  {CECHY[2]:<25} klasa pojazdu (zakodowana liczbowo)
  {CECHY[3]:<25} rodzaj paliwa (zakodowany liczbowo)

SKUTECZNOSC
  MAE = {mae:.3f} L/100km
        Model myli sie srednio o {mae:.2f} L/100km.
        Np. dla auta spalajacego 8.0 L/100km model zwroci
        wartosc z zakresu {8.0-mae:.1f}-{8.0+mae:.1f} L/100km.

  R2  = {r2:.3f}
        Model wyjasnia {r2*100:.1f}% zmiennosci zuzycia paliwa.
        (0.0 = model losowy, 1.0 = model idealny)

WAZNOSC CECH
{"".join(f"  {k:<25} {v*100:.1f}%{chr(10)}" for k, v in fi.items())}
KLASY POJAZDOW ({len(koder_klasy.classes_)} kategorii)
  {chr(10)+"  ".join(koder_klasy.classes_)}

RODZAJE PALIWA
  X = Benzyna 95
  Z = Benzyna 98
  D = Diesel

OGRANICZENIA
  Model przewiduje zuzycie w warunkach testowych.
  Rzeczywiste spalanie moze roznic sie o 10-30% w zaleznosci
  od stylu jazdy, warunkow drogowych i temperatury.

================================================================
"""

with open('model/raport_modelu.txt', 'w', encoding='utf-8') as f:
    f.write(raport)

print("Zapisano:")
print("  model/model.pkl")
print("  model/encoders.pkl")
print("  model/metadata.json")
print("  model/raport_modelu.txt")