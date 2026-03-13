from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

dataset = fetch_openml(name='pc4', version=1, as_frame=True)
X = dataset.data
y = dataset.target

print("=== INFO ===")
print(f"Wiersze: {X.shape[0]}, Cechy: {X.shape[1]}")
print(f"\nTarget:\n{y.value_counts()}")

le = LabelEncoder()
y_enc = le.fit_transform(y)
X_vals = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X_vals, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\nPrzed SMOTE: {sum(y_train==0)} clean, {sum(y_train==1)} buggy")
print(f"Po SMOTE:    {sum(y_train_res==0)} clean, {sum(y_train_res==1)} buggy")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)

print("\n", classification_report(y_test, y_pred, target_names=['clean', 'buggy']))