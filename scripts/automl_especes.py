#!/usr/bin/env python3
# Script pour explorer différents modèles pour la classification d'espèces

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
        logging.FileHandler("automl_especes.log"),
        logging.StreamHandler()
    ]
)

# Désactiver les avertissements
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Charge les données pour la classification d'espèces"""
    try:
        # Chemin vers le fichier de données
        data_path = 'dataset/plantvillage/csv/clean_data_plantvillage_segmented_all_with_features.csv'
        
        # Vérifier si le fichier existe
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Le fichier {data_path} n'existe pas")
            
        # Charger les données
        logging.info(f"Chargement des données depuis {data_path}...")
        df = pd.read_csv(data_path)
        
        # Vérifier les colonnes nécessaires
        required_columns = ['nom_plante', 'nom_maladie']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans les données: {missing_columns}")
            
        logging.info(f"Données chargées avec succès. Taille: {len(df)} échantillons")
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données: {str(e)}")
        raise

def setup_experiment(data, target):
    """Configure l'expérience PyCaret pour la classification d'espèces"""
    try:
        # Colonnes à ignorer (identifiants, chemins d'images, etc.)
        ignore_features = ['Image_Path', 'md5', 'nom_maladie']
        
        # Configuration de base de PyCaret
        logging.info("Configuration de l'expérience PyCaret...")
        exp = setup(
            data=data,
            target=target,
            train_size=0.8,
            session_id=42,
            normalize=True,
            fix_imbalance=True,
            remove_multicollinearity=True,
            ignore_features=ignore_features,
            verbose=False,  # Désactive les sorties verbeuses
            profile=False,
            log_experiment=False,  # Désactiver MLflow pour éviter les erreurs
            experiment_name='especes_classification',
            use_gpu=False
        )
        
        logging.info("Configuration de l'expérience terminée avec succès")
        return exp
    except Exception as e:
        logging.error(f"Erreur lors de la configuration de l'expérience: {str(e)}")
        raise

def train_models():
    """Entraîne et évalue différents modèles pour la classification d'espèces"""
    try:
        # Charger les données
        data = load_data()
        
        # Configurer l'expérience pour la classification d'espèces
        exp = setup_experiment(data, 'nom_plante')
        
        # Liste des modèles à tester (5 modèles)
        models_to_test = [
            'rf',         # Random Forest
            'et',         # Extra Trees
            'lightgbm',   # LightGBM
            'lda',        # Linear Discriminant Analysis
            'knn'         # K-Nearest Neighbors
        ]
        
        # Créer le dossier de résultats s'il n'existe pas
        os.makedirs('results/automl_especes', exist_ok=True)
        
        # Comparaison des modèles avec validation croisée
        logging.info("Démarrage de la comparaison des modèles...")
        best_models = compare_models(
            include=models_to_test,
            n_select=len(models_to_test),  # Sélectionner tous les modèles pour comparaison
            sort='F1',                     # Trier par F1 Score
            cross_validation=True,
            fold=3,                        # Nombre de folds pour la validation croisée
            verbose=False
        )
        
        # Récupérer et afficher les résultats
        results = pull()
        
        # Afficher un résumé des résultats
        print("\n" + "="*100)
        print("RÉSULTATS DE LA COMPARAISON DES MODÈLES (CLASSIFICATION D'ESPÈCES)")
        print("="*100)
        
        # Afficher toutes les colonnes disponibles pour le débogage
        print("\nColonnes disponibles dans les résultats:")
        print(results.columns.tolist())
        
        # Définir les métriques à afficher en fonction de ce qui est disponible
        available_metrics = ['Model']
        
        # Vérifier quelles métriques sont disponibles
        for metric in ['Accuracy', 'F1', 'AUC', 'Recall', 'Prec.', 'Time']:
            if metric in results.columns:
                available_metrics.append(metric)
        
        # Afficher les métriques disponibles
        print("\nMétriques disponibles:", available_metrics)
        print("\nRésultats des modèles:")
        print(results[available_metrics].round(4).to_string())
        
        # Déterminer la métrique à utiliser pour le classement
        score_metric = 'F1' if 'F1' in results.columns else 'Accuracy'
        
        # Trier les résultats par la métrique choisie
        results_sorted = results.sort_values(by=score_metric, ascending=False)
        
        # Afficher le meilleur modèle
        best_model_name = results_sorted.iloc[0]['Model']
        best_score = results_sorted.iloc[0][score_metric]
        print("\n" + "-"*100)
        print(f"MEILLEUR MODÈLE: {best_model_name} ({score_metric}: {best_score:.4f})")
        print("-"*100 + "\n")
        
        # Sauvegarder les résultats détaillés
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'results/automl_especes/model_comparison_{timestamp}.csv'
        results.to_csv(results_file, index=False)
        logging.info(f"Résultats détaillés sauvegardés dans {results_file}")
        
        # Sauvegarder le meilleur modèle
        best_model = best_models[0]
        model_file = f'results/automl_especes/best_model_{timestamp}.pkl'
        save_model(best_model, model_file)
        logging.info(f"Meilleur modèle sauvegardé dans {model_file}")
        
        # Créer un résumé des résultats
        summary = {
            'date_execution': timestamp,
            'meilleur_modele': best_model_name,
            'score_metrique': score_metric,
            'valeur_score': float(best_score),
            'fichier_resultats': results_file,
            'fichier_modele': model_file,
            'metriques': results[available_metrics].to_dict('records')
        }
        
        # Sauvegarder le résumé
        summary_file = f'results/automl_especes/summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=4)
        
        logging.info(f"Résumé des résultats sauvegardé dans {summary_file}")
        
        return best_model, summary
        
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement des modèles: {str(e)}")
        raise

def main():
    try:
        logging.info("Début de l'exploration AutoML pour la classification d'espèces")
        start_time = datetime.now()
        
        # Entraîner et évaluer les modèles
        best_model, summary = train_models()
        
        # Afficher un résumé des résultats
        print("\n" + "#" * 100)
        print("RÉSUMÉ DE L'ENTRAÎNEMENT - CLASSIFICATION D'ESPÈCES")
        print("#" * 100)
        print(f"Date d'exécution: {summary['date_execution']}")
        print(f"Meilleur modèle: {summary['meilleur_modele']}")
        print(f"Métrique: {summary['score_metrique']}")
        print(f"Score: {summary['valeur_score']:.4f}")
        print(f"Fichier des résultats: {summary['fichier_resultats']}")
        print(f"Fichier du modèle: {summary['fichier_modele']}")
        print("\nClassement des modèles:")
        
        # Afficher le classement des modèles
        for i, model in enumerate(summary['metriques'], 1):
            print(f"{i}. {model['Model']} - F1: {model['F1']:.4f}, Précision: {model['Precision']:.4f}, Rappel: {model['Recall']:.4f}")
        
        # Calculer et afficher le temps d'exécution
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "=" * 100)
        print(f"Temps d'exécution total: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print("=" * 100 + "\n")
        logging.info("Exploration AutoML terminée avec succès")
        
    except Exception as e:
        logging.error(f"Une erreur est survenue: {str(e)}")
        raise

if __name__ == "__main__":
    main()
