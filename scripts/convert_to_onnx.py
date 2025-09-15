#!/usr/bin/env python3
# Script pour convertir le modèle en format ONNX

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json
import os

def convert_model():
    # Chemins des fichiers
    model_path = 'results/test_models/tuned_rf_model.joblib'
    metadata_path = 'results/test_models/model_metadata.json'
    onnx_path = 'results/test_models/model.onnx'
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Charger le modèle
    print("Chargement du modèle...")
    model = joblib.load(model_path)
    
    # Charger les métadonnées pour obtenir les caractéristiques
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Définir le type d'entrée attendu par le modèle
    n_features = len(metadata.get('features', []))
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convertir le modèle en ONNX
    print("Conversion en ONNX...")
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12
    )
    
    # Sauvegarder le modèle ONNX
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"Modèle converti avec succès et sauvegardé dans {onnx_path}")
    
    # Vérifier la taille du fichier
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Taille du fichier ONNX: {size_mb:.2f} MB")

if __name__ == "__main__":
    convert_model()
