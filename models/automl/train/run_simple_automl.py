import pandas as pd
import numpy as np
from pycaret.classification import *
import logging
from datetime import datetime
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automl_pipeline.log"),
        logging.StreamHandler()
    ]
)

def load_data(csv_path, target_col):
    """Charge les données et prépare la cible"""
    logging.info(f"Chargement des données depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sélection des caractéristiques (toutes les colonnes numériques sauf la cible)
    features = [col for col in df.select_dtypes(include=np.number).columns 
                if col != target_col and not col.startswith('Unnamed')]
    
    # Préparation des données
    data = df[features + [target_col]].copy()
    data = data.rename(columns={target_col: 'target'})
    
    logging.info(f"Données chargées : {len(data)} échantillons, {len(features)} caractéristiques")
    return data, features

def main():
    # Configuration
    TARGET = 'nom_plante'  # Commençons par l'analyse des espèces
    CSV_PATH = 'dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv'
    OUTPUT_DIR = 'results/automl'
    
    # Création du dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Chargement des données
    data, features = load_data(CSV_PATH, TARGET)
    
    try:
        # Configuration de l'environnement PyCaret
        logging.info("Configuration de l'environnement PyCaret...")
        exp = setup(
            data=data,
            target='target',
            train_size=0.8,
            session_id=42,  # Pour la reproductibilité
            normalize=True,
            normalize_method='robust',
            feature_selection=True,
            feature_selection_method='classic',
            feature_selection_estimator='lightgbm',
            remove_multicollinearity=True,
            multicollinearity_threshold=0.9,
            fix_imbalance=True,
            verbose=True,
            experiment_name=f'automl_{TARGET}_{datetime.now().strftime("%Y%m%d_%H%M")}'
        )
        
        # Comparaison des modèles
        logging.info("Comparaison des modèles...")
        best_models = compare_models(
            include=['lr', 'knn', 'dt', 'rf', 'lightgbm'],
            n_select=3,
            sort='F1',
            fold=5,
            round=4,
            verbose=False
        )
        
        # Sauvegarde du meilleur modèle
        best_model = best_models[0]
        save_model(best_model, f'{OUTPUT_DIR}/best_model_{TARGET}')
        logging.info(f"Meilleur modèle sauvegardé : {best_model.__class__.__name__}")
        
        # Évaluation du modèle
        logging.info("Évaluation du modèle...")
        evaluate_model(best_model)
        
        logging.info(f"Analyse terminée avec succès pour la cible : {TARGET}")
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution du pipeline AutoML: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
