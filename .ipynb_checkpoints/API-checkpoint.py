from fastapi import FastAPI
import joblib
import pandas as pd

# Charger le modèle sauvegardé
model = joblib.load("xgboost_model.pkl")

app = FastAPI()

@app.post("/predict")
def predict(features: dict):
    # Convertir les caractéristiques en DataFrame
    features_df = pd.DataFrame([features])

    # Ajouter les colonnes manquantes
    for col in X_train.columns:
        if col not in features_df.columns:
            features_df[col] = 0

    # Réorganiser les colonnes
    features_df = features_df[X_train.columns]

    # Faire la prédiction
    prediction = model.predict(features_df)
    return {"predicted_price": prediction[0]}
