#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal pour exécuter le pipeline de classification des maladies des plantes.

Exemple d'utilisation:
    python run_pipeline.py --data_dir dataset/plantvillage/images --batch_size 32 --epochs 20
"""

import os
import sys
import argparse
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

try:
    from scripts.plantvillage_classifier import (
        Config,
        PlantVillageDataLoader,
        PlantDiseaseClassifier
    )
    from scripts.utils import Logger
except ImportError:
    print("Erreur: Impossible d'importer les modules requis.")
    print("Assurez-vous d'installer les dépendances avec: pip install -r requirements.txt")
    sys.exit(1)

def parse_arguments():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Exécute le pipeline de classification des maladies des plantes."
    )
    
    # Arguments principaux
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/plantvillage/images",
        help="Chemin vers le dossier contenant les images classées par classe."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Taille des lots pour l'entraînement et l'évaluation."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Nombre d'époques d'entraînement."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Taux d'apprentissage initial."
    )
    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=1e-5,
        help="Taux d'apprentissage pour le fine-tuning."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Dossier de sortie pour les résultats."
    )
    
    return parser.parse_args()

def main():
    """Fonction principale pour exécuter le pipeline."""
    # Parser les arguments
    args = parse_arguments()
    
    # Initialiser la configuration
    config = Config()
    
    # Mettre à jour la configuration avec les arguments
    config.image_dir = Path(args.data_dir)
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.initial_learning_rate = args.learning_rate
    config.fine_tune_learning_rate = args.fine_tune_lr
    
    # Mettre à jour le dossier de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = ensure_dir(Path(args.output_dir) / f"plantvillage_{timestamp}")
    config.model_dir = ensure_dir(config.output_dir / "models")
    config.logs_dir = ensure_dir(config.output_dir / "logs")
    config.plots_dir = ensure_dir(config.output_dir / "plots")
    config.model_path = config.model_dir / "best_model.h5"
    config.metrics_file = config.output_dir / "metrics.json"
    
    # Initialiser le logger
    logger = Logger(str(config.output_dir / 'pipeline.log'))
    
    try:
        # 1. Chargement des données
        logger.info("Étape 1/4 - Chargement des données...")
        data_loader = PlantVillageDataLoader(config)
        train_generator, val_generator, test_generator, class_names = data_loader.load_data()
        
        # Mettre à jour le nombre de classes dans la configuration
        config.num_classes = len(class_names)
        
        # Sauvegarder la configuration
        config.save()
        
        # 2. Initialisation du modèle
        logger.info("Étape 2/4 - Initialisation du modèle...")
        model = PlantDiseaseClassifier(config)
        
        # 3. Entraînement
        logger.info("Étape 3/4 - Début de l'entraînement...")
        history = model.train(train_generator, val_generator)
        
        # 4. Évaluation
        logger.info("Étape 4/4 - Évaluation du modèle...")
        metrics = model.evaluate(test_generator)
        
        # Afficher les métriques
        logger.info("\n=== Métriques d'évaluation ===")
        for metric, value in metrics.items():
            if metric not in ['confusion_matrix', 'classification_report']:
                logger.info(f"{metric}: {value:.4f}")
        
        # 5. Explications (optionnel)
        if len(test_generator.filenames) > 0:
            logger.info("Génération des explications (Grad-CAM, SHAP, LIME)...")
            num_samples = min(3, len(test_generator.filenames))
            sample_indices = np.random.choice(len(test_generator.filenames), num_samples, replace=False)
            X_sample = np.array([test_generator[i][0][0] for i in sample_indices])
            model.explain_predictions(X_sample, class_names)
        
        logger.info("\n=== Pipeline terminé avec succès ===")
        logger.info(f"Résultats enregistrés dans: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
