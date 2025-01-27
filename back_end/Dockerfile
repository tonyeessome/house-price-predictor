# Étape 1 : Utiliser une image de base Python
FROM python:3.8-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier les fichiers nécessaires dans le conteneur
COPY app.py requirements.txt xgboost_retrained_model.json ./

# Étape 4 : Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port sur lequel Flask tourne (par défaut 5000)
EXPOSE 5001

# Étape 6 : Définir la commande pour lancer le backend
CMD ["python", "app.py"]
