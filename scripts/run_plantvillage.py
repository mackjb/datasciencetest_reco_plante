#!/usr/bin/env python3
"""
Script principal pour l'entraînement et l'évaluation du modèle PlantVillage
"""
import os
import sys
import argparse
from pathlib import Path

# Ajout du répertoire parent au chemin Python
sys.path.append(str(Path(__file__).parent.parent))

from scripts.config_plantvillage import *
from scripts.plantvillage_classifier import PlantDiseaseClassifier, PlantVillageDataLoader

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Classification des maladies de plantes avec PlantVillage')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'explain'], default='train',
                      help='Mode d\'exécution: train (entraînement), evaluate (évaluation), explain (explications)')
    parser.add_argument('--model-dir', type=str, default=None,
                      help='Chemin vers le modèle existant (pour les modes evaluate et explain)')
    parser.add_argument('--no-augmentation', action='store_true',
                      help='Désactive l\'augmentation des données')
    args = parser.parse_args()
    
    # Initialisation du chargeur de données
    print("Chargement des données...")
    data_loader = PlantVillageDataLoader(
        image_dir=IMAGE_DIR,
        label_csv=LABEL_CSV,
        img_size=IMG_SIZE,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Chargement des données
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_data()
    
    # Mise à jour des noms de classes dans la configuration
    CLASS_NAMES = data_loader.class_names
    
    # Initialisation ou chargement du modèle
    if args.model_dir:
        print(f"\nChargement du modèle depuis {args.model_dir}")
        classifier = PlantDiseaseClassifier.load_model(Path(args.model_dir) / 'config.json')
    else:
        config = {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'fine_tune_layers': FINE_TUNE_LAYERS,
            'output_dir': str(OUTPUT_DIR),
            'model_path': str(MODEL_DIR / 'best_model.h5'),
            'logs_dir': str(LOGS_DIR),
            'plots_dir': str(PLOTS_DIR),
            'class_names': CLASS_NAMES,
            'num_classes': len(CLASS_NAMES)
        }
        classifier = PlantDiseaseClassifier(config, len(CLASS_NAMES))
    
    # Exécution en fonction du mode
    if args.mode == 'train':
        # Entraînement du modèle
        print("\nDémarrage de l'entraînement...")
        history = classifier.train(
            (X_train, y_train),
            (X_val, y_val),
            use_augmentation=not args.no_augmentation
        )
        
        # Sauvegarde du modèle
        classifier.save_model()
        
        # Affichage des courbes d'apprentissage
        classifier.plot_training_history(history)
        
        # Évaluation sur l'ensemble de test
        print("\nÉvaluation sur l'ensemble de test...")
        classifier.evaluate(X_test, y_test)
        
    elif args.mode == 'evaluate':
        # Évaluation du modèle
        print("\nÉvaluation du modèle...")
        metrics = classifier.evaluate(X_test, y_test)
        
    elif args.mode == 'explain':
        # Génération d'explications
        print("\nGénération d'explications...")
        
        # Sélection d'un échantillon aléatoire pour l'explication
        num_samples = min(3, len(X_test))
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_sample = X_test[sample_indices]
        
        # Génération des explications
        classifier.explain_predictions(X_sample, CLASS_NAMES)
        
        print("\nExplications générées avec succès!")

if __name__ == "__main__":
    main()
