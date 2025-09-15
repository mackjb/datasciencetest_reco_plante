import pandas as pd
import numpy as np
from pycaret.classification import *
import logging
import os
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("automl_test.log"),
        logging.StreamHandler()
    ]
)

def main():
    try:
        # Charger un plus grand sous-ensemble de données
        logging.info("Chargement des données...")
        df = pd.read_csv('dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv')
        
        # Prendre un échantillon plus petit pour accélérer les tests
        sample = df.sample(2000, random_state=42)  # Réduit à 2000 échantillons pour les tests
        
        # Préparer les données
        features = [col for col in sample.select_dtypes(include=np.number).columns 
                  if col not in ['nom_plante', 'nom_maladie', 'Est_Saine']]
        
        data = sample[features + ['nom_plante']].copy()
        data = data.rename(columns={'nom_plante': 'target'})
        
        # Créer le dossier de sortie
        os.makedirs('results/test_models', exist_ok=True)
        
        # Configuration simplifiée de PyCaret pour accélérer les tests
        logging.info("Configuration de PyCaret...")
        exp = setup(
            data=data,
            target='target',
            train_size=0.8,
            session_id=42,
            normalize=True,  # Normalisation de base
            normalize_method='zscore',  # Plus rapide que 'robust'
            feature_selection=False,  # Désactiver la sélection de caractéristiques
            remove_multicollinearity=False,  # Désactiver la suppression de la multicolinéarité
            fix_imbalance=False,  # Désactiver la gestion du déséquilibre
            fold_strategy='kfold',  # Plus rapide que 'stratifiedkfold'
            fold=2,  # Réduire le nombre de folds
            verbose=True,
            log_experiment=False,
            experiment_name=f'test_models_{datetime.now().strftime("%Y%m%d_%H%M")}'
        )
        
        # Modèles les plus rapides pour les tests initiaux
        models_to_compare = [
            'lr',  # Régression logistique (le plus rapide)
            'dt'   # Arbre de décision (rapide et interprétable)
        ]
        
        # Comparaison des modèles
        logging.info("Comparaison des modèles...")
        best_models = compare_models(
            include=models_to_compare,
            n_select=1,  # Sélectionner uniquement le meilleur modèle
            sort='Accuracy',  # Trier par précision pour la simplicité
            fold=2,      # 2-fold cross-validation pour accélérer
            round=4,     # 4 décimales pour les métriques
            verbose=True
        )
        
        # Sauvegarder les meilleurs modèles
        for i, model in enumerate(best_models):
            model_name = model.__class__.__name__
            save_model(model, f'results/test_models/best_model_{i+1}_{model_name}')
            logging.info(f"Modèle {i+1} sauvegardé: {model_name}")
        
        # Afficher le meilleur modèle
        best_model = best_models[0]
        logging.info(f"\n🎉 Meilleur modèle: {best_model.__class__.__name__}")
        
        # Évaluation du meilleur modèle
        logging.info("\n📊 Évaluation du meilleur modèle...")
        evaluate_model(best_model)
        
        # Prédictions sur l'ensemble de test
        logging.info("\n🔮 Génération des prédictions...")
        predictions = predict_model(best_model)
        
        # Sauvegarder les prédictions
        predictions.to_csv('results/test_models/predictions.csv', index=False)
        logging.info("Prédictions sauvegardées dans results/test_models/predictions.csv")
        
    except Exception as e:
        logging.error(f"\n❌ Erreur: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
