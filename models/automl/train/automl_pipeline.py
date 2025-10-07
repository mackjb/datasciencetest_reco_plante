#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline AutoML pour la classification d'espèces et de maladies de plantes
"""

import pandas as pd
import json
import logging
from pathlib import Path
from pycaret.classification import *
import numpy as np
import warnings
from datetime import datetime
from src.config import load_config

# Chargement config centrale YAML (optionnelle)
YAML_CFG = load_config()

# Chargement de la configuration
CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'automl_config.json'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Configuration des chemins (priorité au JSON, sinon YAML, sinon défaut)
yaml_results = Path(YAML_CFG.get('paths', {}).get('results_dir', 'results'))
default_output = yaml_results / 'automl'
OUTPUT_DIR = Path(CONFIG.get('output', {}).get('directory', str(default_output)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration du logging (log dans OUTPUT_DIR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'automl_pipeline.log'),
        logging.StreamHandler()
    ]
)

# Désactivation des warnings
warnings.filterwarnings('ignore')

def load_data(config):
    """Charge et prépare les données"""
    data_config = config['data']
    target_type = data_config['target_type']
    csv_path = Path(data_config['csv_path'])
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de données non trouvé : {csv_path}")
    
    logging.info(f"Chargement des données depuis {csv_path} pour la cible: {target_type}")
    
    # Lecture des données avec gestion des guillemets
    df = pd.read_csv(csv_path, quotechar='"', quoting=1)
    
    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Sélection de la cible
    if target_type == 'espece':
        # Utilisation de la colonne 'nom_plante' comme cible
        df['target'] = df['nom_plante'].str.strip()
    else:  # maladie
        # Utilisation de la colonne 'nom_maladie' comme cible
        df['target'] = df['nom_maladie'].str.strip()
    
    # Suppression des colonnes non numériques inutiles
    features_to_drop = ['nom_plante', 'nom_maladie', 'Est_Saine', 'Image_Path', 'md5']
    features = [col for col in df.columns if col not in features_to_drop and col != 'target']
    
    logging.info(f"Données chargées : {len(df)} échantillons, {len(features)} caractéristiques")
    return df[features + ['target']], features

def run_automl(data, features, config):
    """Exécute le pipeline AutoML avec PyCaret"""
    target_type = config['data']['target_type']
    logging.info(f"Démarrage du pipeline AutoML pour la cible: {target_type}")
    
    try:
        # Configuration de l'expérience PyCaret
        logging.info("Configuration de l'environnement PyCaret...")
        # Configuration minimale de PyCaret 3.3.2
        exp = setup(
            data=data,
            target='target',
            train_size=1 - config['data']['test_size'],
            session_id=42,  # Pour la reproductibilité
            normalize=config['preprocessing']['normalize'],
            normalize_method=config['preprocessing']['normalize_method'],
            feature_selection=config['preprocessing']['feature_selection'],
            feature_selection_method='classic',
            feature_selection_estimator='lightgbm',
            remove_multicollinearity=config['preprocessing'].get('remove_multicollinearity', True),
            multicollinearity_threshold=config['preprocessing'].get('multicollinearity_threshold', 0.9),
            fix_imbalance=config['preprocessing']['fix_imbalance'],
            verbose=True,  # Afficher les logs détaillés
            experiment_name=f'automl_{target_type}_{datetime.now().strftime("%Y%m%d_%H%M")}'
        )
        
        # Comparaison des modèles
        logging.info("Comparaison des modèles...")
        best_models = compare_models(
            include=config['models']['include'],
            n_select=config['models'].get('n_best_models', 3),
            sort=config['models']['optimize_metric'],
            fold=config['models'].get('cv_folds', 5),
            round=4,
            verbose=False
        )
        
        # Optimisation du meilleur modèle
        logging.info("Optimisation du meilleur modèle...")
        tuned_model = tune_model(
            best_models[0],
            optimize=config['models']['optimize_metric'].lower(),
            n_iter=config['optimization']['n_iter'],
            search_library=config['optimization']['search_library'],
            search_algorithm=config['optimization']['search_algorithm'],
            early_stopping=config['optimization']['early_stopping'],
            choose_better=True,
            verbose=False
        )
        
        # Évaluation finale
        logging.info("Évaluation du modèle final...")
        evaluate_model(tuned_model)
        
        return tuned_model
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution du pipeline AutoML: {str(e)}")
        raise

def main():
    try:
        logging.info("Démarrage du pipeline AutoML")
        
        # Chargement des données
        data, features = load_data(CONFIG)
        
        # Exécution du pipeline AutoML
        best_model = run_automl(data, features, CONFIG)
        
        # Sauvegarde du modèle si configuré
        if CONFIG['output'].get('save_model', True):
            model_path = OUTPUT_DIR / f'best_model_{CONFIG["data"]["target_type"]}'
            save_model(best_model, str(model_path))
            logging.info(f"Modèle sauvegardé dans : {model_path}.pkl")
            
            # Sauvegarde des prédictions si configuré
            if CONFIG['output'].get('save_predictions', True):
                predictions = predict_model(best_model, data=data)
                pred_path = OUTPUT_DIR / f'predictions_{CONFIG["data"]["target_type"]}.csv'
                predictions.to_csv(pred_path, index=False)
                logging.info(f"Prédictions sauvegardées dans : {pred_path}")
        
        logging.info("Pipeline AutoML terminé avec succès")
        
    except Exception as e:
        logging.error(f"Erreur critique dans le pipeline AutoML: {str(e)}")
        raise

if __name__ == "__main__":
    main()
