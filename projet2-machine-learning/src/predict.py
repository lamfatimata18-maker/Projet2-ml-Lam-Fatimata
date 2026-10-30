import pandas as pd
import joblib
import os

def predict(df: pd.DataFrame) -> list:
    """
    Fonction de prédiction appelée par le correcteur automatique.
    df : mêmes colonnes que flux_train.csv, SANS 'attaque'
    Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne.
    """
    # Déterminer le chemin vers le modèle sauvegardé à la racine du projet
    model_path = os.path.join(os.path.dirname(__file__), '../models/model.joblib')
    
    # Charger le modèle entraîné (Pipeline complet incluant le prétraitement)
    model = joblib.load(model_path)
    
    # Effectuer les prédictions
    predictions = model.predict(df)
    
    # Retourner les résultats sous forme de liste
    return predictions.tolist()