import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib
import os

print("Chargement des données...")
df = pd.read_csv('../data/flux_train.csv')

X = df.drop(columns=['attaque'])
y = df['attaque']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_f1 = 0
best_name = ""

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    print(f"\n--- Entraînement de : {name} ---")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    f1_attaque = f1_score(y_val, y_pred, pos_label=1)
    print(f"F1-score (attaque) pour {name} : {f1_attaque:.4f}")
    print(classification_report(y_val, y_pred))
    
    if f1_attaque > best_f1:
        best_f1 = f1_attaque
        best_model = clf
        best_name = name

print(f"\nMeilleur modèle retenu : {best_name} avec un F1-score de {best_f1:.4f}")

print("\nRé-entraînement sur l'ensemble des données...")
best_model.fit(X, y)

os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/model.joblib')
print("Modèle sauvegardé avec succès dans models/model.joblib !")