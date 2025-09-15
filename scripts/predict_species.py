#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PlantSpeciesClassifier:
    """
    Classe pour charger et utiliser le modèle de classification d'espèces végétales
    """
    def __init__(self, model_path='results/test_models/tuned_rf_model.joblib', 
                 metadata_path='results/test_models/model_metadata.json'):
        """
        Initialise le classifieur avec le modèle et les métadonnées
        """
        self.model = None
        self.metadata = None
        self.features = None
        self.classes = None
        
        # Charger le modèle et les métadonnées
        self.load_model(model_path, metadata_path)
    
    def load_model(self, model_path, metadata_path):
        """Charge le modèle et les métadonnées"""
        try:
            # Charger le modèle
            self.model = joblib.load(model_path)
            
            # Charger les métadonnées
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Extraire les informations utiles
            self.features = self.metadata.get('features', [])
            self.classes = list(self.metadata.get('target_mapping', {}).keys())
            
            logging.info(f"Modèle chargé avec succès. Précision: {self.metadata.get('accuracy', 0):.2%}")
            logging.info(f"Classes disponibles: {', '.join(self.classes) if self.classes else 'Non spécifiées'}")
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Prétraite une image pour la prédiction
        (À implémenter selon les caractéristiques extraites lors de l'entraînement)
        """
        try:
            # Ici, vous devrez implémenter le même prétraitement que celui utilisé pour l'entraînement
            # Ceci est un exemple simplifié
            features = {}
            
            # Exemple: dimensions de l'image
            with Image.open(image_path) as img:
                width, height = img.size
                features['width_img'] = width
                features['height_img'] = height
            
            # Convertir en DataFrame avec les mêmes colonnes que pendant l'entraînement
            df = pd.DataFrame([features])
            
            # S'assurer que toutes les caractéristiques nécessaires sont présentes
            for feat in self.features:
                if feat not in df.columns:
                    df[feat] = 0  # Valeur par défaut
            
            return df[self.features]
            
        except Exception as e:
            logging.error(f"Erreur lors du prétraitement de l'image: {str(e)}")
            raise
    
    def predict(self, image_path):
        """
        Prédit l'espèce végétale à partir d'une image
        """
        try:
            # Prétraiter l'image
            features_df = self.preprocess_image(image_path)
            
            # Faire la prédiction
            probabilities = self.model.predict_proba(features_df)[0]
            predicted_class_idx = self.model.predict(features_df)[0]
            
            # Récupérer le nom de la classe prédite
            predicted_class = self.classes[predicted_class_idx] if self.classes else f"Classe {predicted_class_idx}"
            
            # Créer un dictionnaire des probabilités par classe
            class_probabilities = {}
            for i, prob in enumerate(probabilities):
                class_name = self.classes[i] if self.classes else f"Classe {i}"
                class_probabilities[class_name] = float(prob)
            
            # Trier les probabilités par ordre décroissant
            sorted_probabilities = dict(sorted(
                class_probabilities.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(probabilities[predicted_class_idx]),
                'probabilities': sorted_probabilities,
                'metadata': {
                    'model_accuracy': self.metadata.get('accuracy'),
                    'model_type': self.metadata.get('model_type'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction: {str(e)}")
            raise

def main():
    # Exemple d'utilisation
    try:
        # Initialiser le classifieur
        classifier = PlantSpeciesClassifier()
        
        # Exemple de prédiction (remplacez par le chemin de votre image)
        image_path = "chemin/vers/votre/image.jpg"
        
        if os.path.exists(image_path):
            # Faire la prédiction
            result = classifier.predict(image_path)
            
            # Afficher les résultats
            print("\n" + "="*50)
            print(f"Image analysée: {image_path}")
            print(f"Espèce prédite: {result['predicted_class']}")
            print(f"Confiance: {result['confidence']:.2%}")
            print("\nProbabilités par classe:")
            for class_name, prob in result['probabilities'].items():
                print(f"- {class_name}: {prob:.2%}")
            print("="*50 + "\n")
            
        else:
            print(f"Le fichier {image_path} n'existe pas.")
            print("Veuillez spécifier un chemin d'image valide.")
    
    except Exception as e:
        logging.error(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main()
