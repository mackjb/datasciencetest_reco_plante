#!/usr/bin/env python3
# Script pour tester le modèle avec une image exemple

import os
import sys
from scripts.use_model import SimpleModel

def main():
    # Vérifier si un chemin d'image est fourni en argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Chemin par défaut pour une image exemple
        image_path = "data/examples/test_plant.jpg"
        
        # Créer le dossier d'exemple s'il n'existe pas
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Message d'aide si l'image exemple n'existe pas
        if not os.path.exists(image_path):
            print("Aucune image spécifiée. Utilisation :")
            print(f"  {sys.argv[0]} chemin/vers/votre/image.jpg")
            print("\nOu placez une image dans data/examples/ et nommez-la 'test_plant.jpg'")
            return
    
    # Initialiser et utiliser le modèle
    try:
        print("Chargement du modèle...")
        model = SimpleModel()
        
        print(f"\nAnalyse de l'image: {image_path}")
        result = model.predict(image_path)
        
        # Afficher les résultats
        print("\n" + "="*50)
        print(f"Résultat de la prédiction:")
        print(f"- Espèce prédite: Classe {result['predicted_class']}")
        print(f"- Niveau de confiance: {result['confidence']:.2%}")
        
        print("\nDétails des probabilités:")
        for class_id, prob in result['probabilities'].items():
            print(f"- Classe {class_id}: {prob:.2%}")
        
        print("\nCaractéristiques extraites (premières valeurs):")
        features = list(result['features'].items())[:5]  # Afficher seulement les 5 premières
        for feat, value in features:
            print(f"- {feat}: {value:.4f}")
        print("...")
        
        print("\nNote: Cette version utilise une prédiction simulée basée sur l'importance des caractéristiques.")
        print("Pour des résultats réels, utilisez le modèle complet avec joblib.")
        print("="*50)
        
    except Exception as e:
        print(f"\nErreur lors de l'analyse de l'image: {str(e)}")
        print("Assurez-vous que le chemin de l'image est correct et que le modèle est disponible.")

if __name__ == "__main__":
    main()
