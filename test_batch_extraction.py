#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour vérifier l'extraction des caractéristiques sur un batch d'images
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Importer les fonctions nécessaires de my_feature_importance.py
from my_feature_importance import (
    extract_features_from_image, 
    PROJECT_ROOT, 
    DATA_DIR, 
    CSV_FILE,
    FEATURE_COLUMNS
)

def test_batch_extraction(batch_size=10):
    """Test l'extraction des caractéristiques sur un batch d'images"""
    # Charger le DataFrame
    print(f"Chargement du CSV depuis {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # Prendre un échantillon aléatoire
    sample_df = df.sample(batch_size)
    print(f"Échantillon de {batch_size} images à tester")
    
    # Compter le nombre de colonnes de caractéristiques attendues
    print(f"Nombre de colonnes de caractéristiques attendues: {len(FEATURE_COLUMNS)}")
    
    # Statistiques
    success_count = 0
    features_found = {}
    
    # Tester l'extraction sur chaque image
    for idx, row in tqdm(sample_df.iterrows(), total=batch_size, desc="Extraction"):
        img_path = row['filepath']
        
        # Résoudre le chemin complet
        if not Path(img_path).is_absolute():
            if img_path.startswith('dataset'):
                img_path = PROJECT_ROOT / img_path
            else:
                img_path = PROJECT_ROOT / img_path
        
        # Vérifier si le fichier existe
        if not Path(img_path).exists():
            print(f"  - Fichier non trouvé: {img_path}")
            continue
            
        # Extraire les caractéristiques
        features = extract_features_from_image(img_path)
        
        if features is None:
            print(f"  - Extraction échouée pour {img_path}")
            continue
            
        # Compter les succès
        success_count += 1
        
        # Enregistrer les caractéristiques trouvées
        for key in features.keys():
            if key not in features_found:
                features_found[key] = 0
            features_found[key] += 1
    
    # Afficher les résultats
    print(f"\nRésultats: {success_count}/{batch_size} extractions réussies ({success_count/batch_size*100:.2f}%)")
    
    if success_count > 0:
        print(f"\nCaractéristiques extraites ({len(features_found)}):")
        for key, count in sorted(features_found.items()):
            print(f"  - {key}: trouvée dans {count}/{success_count} images ({count/success_count*100:.2f}%)")
    
    # Vérifier si toutes les colonnes attendues sont présentes
    if features_found:
        missing_cols = set(FEATURE_COLUMNS) - set(features_found.keys())
        if missing_cols:
            print(f"\nColonnes manquantes ({len(missing_cols)}):")
            for col in sorted(missing_cols):
                print(f"  - {col}")
        else:
            print("\nToutes les colonnes attendues ont été extraites correctement!")

if __name__ == "__main__":
    batch_size = 20  # Tester sur 20 images aléatoires
    test_batch_extraction(batch_size)
