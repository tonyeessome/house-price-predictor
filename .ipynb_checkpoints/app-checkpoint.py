from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Charger le modèle sauvegardé
model = joblib.load("xgboost.pkl")

# Colonnes utilisées par le modèle
X_train_columns = ["location_id", "baths", "bedrooms", "Area_in_Square_Meters"]

# Initialiser l'application Flask
app = Flask(__name__)

# Activer le support CORS
CORS(app)

# Endpoint racine
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bienvenue dans l'API de prédiction des prix immobiliers."})

# Endpoint de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données JSON envoyées
        data = request.get_json()

        # Convertir les données en DataFrame
        features_df = pd.DataFrame([data])

        # Ajouter les colonnes manquantes
        for col in X_train_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        # Réorganiser les colonnes
        features_df = features_df[X_train_columns]

        # Faire la prédiction
        prediction = model.predict(features_df)
        return jsonify({"predicted_price": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
