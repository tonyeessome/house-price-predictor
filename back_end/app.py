from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import pandas as pd

# Initialiser l'application Flask
app = Flask(__name__)
CORS(app)

# Charger le modèle XGBoost sauvegardé au format JSON
model = xgb.Booster()
model.load_model("xgboost_retrained_model.json")

# Colonnes utilisées par le modèle
X_train_columns = [
    "location_id", "property_type", "baths", "bedrooms", "area_in_square_meters"
]

# Endpoint racine
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bienvenue dans l'API de prédiction des prix immobiliers."})

# Endpoint de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données JSON envoyées par le frontend
        data = request.get_json()

        # Convertir les données utilisateur en DataFrame
        user_input = pd.DataFrame([data])

        # Ajouter les colonnes manquantes avec des valeurs par défaut
        default_values = {col: 0 for col in X_train_columns}
        for col, default in default_values.items():
            if col not in user_input.columns:
                user_input[col] = default

        # Réorganiser les colonnes dans l'ordre attendu par le modèle
        user_input = user_input[X_train_columns]

        # Convertir le DataFrame en DMatrix pour XGBoost
        dmatrix = xgb.DMatrix(user_input)

        # Faire la prédiction
        prediction = model.predict(dmatrix)
        return jsonify({"predicted_price": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


#comment