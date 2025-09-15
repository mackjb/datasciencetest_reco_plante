
# Exemple de code pour utiliser le modèle sauvegardé
import json
import numpy as np

# Charger les informations du modèle
with open('results/test_models/model_simple.json', 'r') as f:
    model_info = json.load(f)

# Afficher les informations du modèle
print(f"Modèle RandomForest avec {model_info['n_estimators']} arbres")
print(f"Classes: {model_info['classes_']}")
print(f"Caractéristiques: {model_info['features']}")
print(f"Importance des caractéristiques: {model_info['feature_importances_']}")

# Pour utiliser ces informations, vous devrez implémenter la logique de prédiction
# ou utiliser directement le modèle original si possible.
