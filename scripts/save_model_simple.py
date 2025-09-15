#!/usr/bin/env python3
# Script pour sauvegarder le modèle dans un format simple (JSON)

import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def save_model_simple():
    # Chemins des fichiers
    model_path = 'results/test_models/tuned_rf_model.joblib'
    metadata_path = 'results/test_models/model_metadata.json'
    output_path = 'results/test_models/model_simple.json'
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Charger le modèle
        print("Chargement du modèle...")
        model = joblib.load(model_path)
        
        # Vérifier que c'est bien un RandomForest
        if not isinstance(model, RandomForestClassifier):
            raise ValueError("Le modèle chargé n'est pas un RandomForestClassifier")
        
        # Charger les métadonnées
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extraire les informations importantes du modèle
        model_info = {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'classes_': model.classes_.tolist(),
            'n_classes_': model.n_classes_,
            'n_features_': model.n_features_in_,
            'feature_importances_': model.feature_importances_.tolist(),
            'features': metadata.get('features', [])
        }
        
        # Sauvegarder dans un fichier JSON
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"Modèle sauvegardé dans {output_path}")
        
        # Créer un exemple de code pour charger et utiliser le modèle
        example_code = """
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
"""
        
        # Sauvegarder l'exemple de code
        example_path = 'results/test_models/example_usage.py'
        with open(example_path, 'w') as f:
            f.write(example_code)
            
        print(f"Exemple d'utilisation sauvegardé dans {example_path}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
        raise

if __name__ == "__main__":
    save_model_simple()
