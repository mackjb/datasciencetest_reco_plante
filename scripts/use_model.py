#!/usr/bin/env python3
# Script pour utiliser le modèle à partir du fichier JSON

import json
import numpy as np
import pandas as pd
import os
from PIL import Image
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SimpleModel:
    """Classe pour charger et utiliser le modèle à partir du fichier JSON"""
    
    def __init__(self, model_path='results/test_models/model_simple.json'):
        """Initialise le modèle à partir du fichier JSON"""
        self.model_info = self._load_model(model_path)
        self._validate_model()
        
    def _load_model(self, model_path):
        """Charge les informations du modèle depuis le fichier JSON"""
        try:
            with open(model_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    
    def _validate_model(self):
        """Vérifie que le modèle chargé est valide"""
        required_keys = ['n_estimators', 'classes_', 'features', 'feature_importances_']
        for key in required_keys:
            if key not in self.model_info:
                raise ValueError(f"Clé manquante dans le modèle: {key}")
        
        logging.info(f"Modèle chargé avec succès (RandomForest avec {self.model_info['n_estimators']} arbres)")
        logging.info(f"Classes: {self.model_info['classes_']}")
        logging.info(f"Nombre de caractéristiques: {len(self.model_info['features'])}")
    
    def preprocess_image(self, image_path):
        """
        Prétraite une image pour la prédiction
        (Version simplifiée - à adapter selon vos besoins)
        """
        try:
            features = {}
            
            # Exemple simple: dimensions de l'image
            with Image.open(image_path) as img:
                width, height = img.size
                features['width_img'] = width
                features['height_img'] = height
                
                # Ajouter des valeurs par défaut pour les autres caractéristiques
                for feat in self.model_info['features']:
                    if feat not in features:
                        features[feat] = 0.0
            
            # Créer un DataFrame avec les caractéristiques dans le bon ordre
            df = pd.DataFrame([features])
            
            # Vérifier que toutes les caractéristiques sont présentes
            missing = [f for f in self.model_info['features'] if f not in df.columns]
            if missing:
                logging.warning(f"Caractéristiques manquantes: {missing}")
                for m in missing:
                    df[m] = 0.0
            
            return df[self.model_info['features']]
            
        except Exception as e:
            logging.error(f"Erreur lors du prétraitement de l'image: {str(e)}")
            raise
    
    def predict(self, image_path):
        """
        Fait une prédiction sur une image
        (Version simulée - dans une vraie implémentation, vous utiliseriez le vrai modèle)
        """
        try:
            # Prétraiter l'image
            features = self.preprocess_image(image_path)
            
            # Ici, nous simulons une prédiction basée sur l'importance des caractéristiques
            # Dans une vraie implémentation, vous utiliseriez le vrai modèle
            logging.warning("ATTENTION: Cette version utilise une prédiction simulée basée sur l'importance des caractéristiques")
            
            # Calculer un score basé sur les caractéristiques et leur importance
            scores = []
            for i, (feat, imp) in enumerate(zip(self.model_info['features'], self.model_info['feature_importances_'])):
                score = float(features.iloc[0][feat]) * imp
                scores.append(score)
            
            # Normaliser les scores entre 0 et 1
            total = sum(scores)
            if total > 0:
                scores = [s/total for s in scores]
            
            # Créer un dictionnaire des probabilités par classe
            # (Dans une vraie implémentation, ce serait la sortie du modèle)
            class_probs = {}
            for i, class_id in enumerate(self.model_info['classes_']):
                class_probs[class_id] = scores[i % len(scores)] if scores else 1.0/len(self.model_info['classes_'])
            
            # Trier les classes par probabilité décroissante
            sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'predicted_class': sorted_probs[0][0],
                'confidence': float(sorted_probs[0][1]),
                'probabilities': dict(sorted_probs),
                'features': features.to_dict('records')[0]
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction: {str(e)}")
            raise

def main():
    # Exemple d'utilisation
    try:
        # Initialiser le modèle
        model = SimpleModel()
        
        # Exemple de prédiction (remplacez par le chemin de votre image)
        image_path = "chemin/vers/votre/image.jpg"
        
        if os.path.exists(image_path):
            # Faire la prédiction
            result = model.predict(image_path)
            
            # Afficher les résultats
            print("\n" + "="*50)
            print(f"Image analysée: {image_path}")
            print(f"Espèce prédite: {result['predicted_class']}")
            print(f"Confiance: {result['confidence']:.2%}")
            
            print("\nProbabilités par classe:")
            for class_id, prob in result['probabilities'].items():
                print(f"- Classe {class_id}: {prob:.2%}")
                
            print("\nCaractéristiques extraites:")
            for feat, value in result['features'].items():
                print(f"- {feat}: {value:.4f}")
                
            print("="*50 + "\n")
            
        else:
            print(f"Le fichier {image_path} n'existe pas.")
            print("Veuillez spécifier un chemin d'image valide.")
    
    except Exception as e:
        logging.error(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main()
