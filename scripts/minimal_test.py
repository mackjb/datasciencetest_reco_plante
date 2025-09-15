import pandas as pd
import numpy as np
from pycaret.classification import *
import logging
import os
from datetime import datetime
import json
import joblib

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("minimal_test.log"),
        logging.StreamHandler()
    ]
)

def main():
    try:
        # Charger un petit échantillon de données
        logging.info("Chargement des données...")
        df = pd.read_csv('dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv')
        
        # Équilibrer les classes en prenant le même nombre d'échantillons par classe
        min_samples = df['nom_plante'].value_counts().min()
        sample = df.groupby('nom_plante').apply(lambda x: x.sample(min(min_samples, len(x)), random_state=42)).reset_index(drop=True)
        
        # Préparer les données
        features = [col for col in sample.select_dtypes(include=np.number).columns 
                  if col not in ['nom_plante', 'nom_maladie', 'Est_Saine']]
        
        data = sample[features + ['nom_plante']].copy()
        data = data.rename(columns={'nom_plante': 'target'})
        
        # Configuration avancée de PyCaret avec gestion du déséquilibre
        logging.info("Configuration de PyCaret...")
        exp = setup(
            data=data,
            target='target',
            train_size=0.8,
            session_id=42,
            normalize=True,
            normalize_method='robust',
            fix_imbalance=True,
            fix_imbalance_method='smote',
            fold_strategy='stratifiedkfold',
            fold=3,
            verbose=True
        )
        
        # Créer et évaluer un modèle plus puissant (Random Forest)
        logging.info("Entraînement du modèle Random Forest...")
        rf = create_model('rf', fold=3)
        
        # Afficher les métriques
        logging.info("Métriques du modèle:")
        print(pull())
        
        # Tunez les hyperparamètres du modèle
        tuned_rf = tune_model(rf, optimize='F1', n_iter=10)
        
        # Évaluer le modèle final
        logging.info("Évaluation du modèle final...")
        evaluate_model(tuned_rf)
        
        # Créer le dossier de résultats s'il n'existe pas
        os.makedirs('results/test_models', exist_ok=True)
        
        # Chemin de sauvegarde du modèle
        model_path = 'results/test_models/tuned_rf_model.joblib'
        
        # Sauvegarder le modèle au format Joblib
        joblib.dump(tuned_rf, model_path)
        
        # Récupérer les métriques d'évaluation
        metrics = pull()
        accuracy = metrics.loc['Mean', 'Accuracy']
        
        # Sauvegarder les métadonnées du modèle
        metadata = {
            'model_type': 'RandomForest',
            'accuracy': float(accuracy),  # Convertir en type natif Python
            'features': features,
            'target': 'nom_plante',
            'created_at': datetime.now().isoformat(),
            'metrics': {
                'AUC': float(metrics.loc['Mean', 'AUC']),
                'F1': float(metrics.loc['Mean', 'F1']),
                'Kappa': float(metrics.loc['Mean', 'Kappa']),
                'MCC': float(metrics.loc['Mean', 'MCC'])
            }
        }
        
        # Sauvegarder les métadonnées au format JSON
        metadata_path = 'results/test_models/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Modèle sauvegardé dans {model_path}")
        logging.info(f"Métadonnées sauvegardées dans {metadata_path}")
        logging.info(f"Précision du modèle: {accuracy:.4f}")
        
    except Exception as e:
        logging.error(f"Erreur: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
