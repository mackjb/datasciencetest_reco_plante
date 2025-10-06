#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test de l'extraction des descripteurs de forme
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import random

# Importation des fonctions requises
from my_feature_importance import extract_features_from_image, SHAPE_DESCRIPTOR_FEATURE_COLS
from src.helpers.helpers import PROJECT_ROOT

def test_shape_extraction_single():
    """
    Teste l'extraction des descripteurs de forme sur une seule image.
    """
    # Chemin vers une image existante dans le dataset
    csv_file = Path(PROJECT_ROOT) / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all.csv"
    
    if not csv_file.exists():
        print(f"Fichier CSV introuvable : {csv_file}")
        return
    
    # Charger le CSV et récupérer une ligne aléatoire
    df = pd.read_csv(csv_file)
    random_idx = random.randint(0, len(df) - 1)
    img_path_rel = df.iloc[random_idx]['filepath']
    
    # Résoudre le chemin complet
    img_path_abs = Path(PROJECT_ROOT) / img_path_rel
    
    print(f"Test sur l'image : {img_path_abs}")
    
    # Vérifier que l'image existe
    if not img_path_abs.exists():
        print(f"L'image n'existe pas : {img_path_abs}")
        return
        
    # Extraire les caractéristiques
    features = extract_features_from_image(str(img_path_abs), normalize_pixels=True)
    
    if features is None:
        print("Échec de l'extraction des caractéristiques.")
        return
    
    # Vérifier que les descripteurs de forme sont présents
    shape_features_found = []
    shape_features_missing = []
    
    for shape_feature in SHAPE_DESCRIPTOR_FEATURE_COLS:
        if shape_feature in features:
            shape_features_found.append(shape_feature)
        else:
            shape_features_missing.append(shape_feature)
    
    # Afficher les résultats
    print("\n=== RÉSULTATS DE L'EXTRACTION ===")
    print(f"Nombre total de caractéristiques extraites : {len(features)}")
    print(f"Nombre de descripteurs de forme attendus : {len(SHAPE_DESCRIPTOR_FEATURE_COLS)}")
    print(f"Nombre de descripteurs de forme trouvés : {len(shape_features_found)}")
    
    if shape_features_missing:
        print(f"\n⚠️ {len(shape_features_missing)} DESCRIPTEURS DE FORME MANQUANTS :")
        for missing in shape_features_missing:
            print(f"  - {missing}")
    else:
        print("\n✅ TOUS LES DESCRIPTEURS DE FORME SONT PRÉSENTS")
    
    if shape_features_found:
        print("\n✅ DESCRIPTEURS DE FORME EXTRAITS :")
        for found in shape_features_found:
            print(f"  - {found}: {features[found]}")

def test_batch_shape_extraction():
    """
    Teste l'extraction des descripteurs de forme sur un lot d'images.
    """
    # Charger le CSV
    csv_file = Path(PROJECT_ROOT) / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all.csv"
    
    if not csv_file.exists():
        print(f"Fichier CSV introuvable : {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    
    # Sélectionner 20 images aléatoires
    sample_size = 20
    sample_indices = random.sample(range(len(df)), sample_size)
    df_sample = df.iloc[sample_indices]
    
    print(f"Test sur {sample_size} images aléatoires...")
    
    # Statistiques
    success_count = 0
    shape_features_stats = {feature: 0 for feature in SHAPE_DESCRIPTOR_FEATURE_COLS}
    
    # Tester chaque image
    for idx, row in df_sample.iterrows():
        img_path_rel = row['filepath']
        img_path_abs = Path(PROJECT_ROOT) / img_path_rel
        
        if not img_path_abs.exists():
            print(f"Image introuvable : {img_path_abs}")
            continue
            
        # Extraire les caractéristiques
        features = extract_features_from_image(str(img_path_abs), normalize_pixels=True)
        
        if features is None:
            print(f"Échec de l'extraction pour {img_path_abs}")
            continue
            
        success_count += 1
        
        # Vérifier les descripteurs de forme
        for shape_feature in SHAPE_DESCRIPTOR_FEATURE_COLS:
            if shape_feature in features:
                shape_features_stats[shape_feature] += 1
    
    # Afficher les résultats
    print("\n=== RÉSULTATS DU TEST BATCH ===")
    print(f"Succès d'extraction : {success_count}/{sample_size} ({success_count/sample_size*100:.1f}%)")
    
    print("\nTaux de présence des descripteurs de forme :")
    for feature, count in shape_features_stats.items():
        print(f"  - {feature}: {count}/{success_count} ({count/success_count*100 if success_count > 0 else 0:.1f}%)")

if __name__ == "__main__":
    print("=== TEST EXTRACTION DES DESCRIPTEURS DE FORME ===\n")
    print("1. Test sur une seule image")
    test_shape_extraction_single()
    
    print("\n2. Test sur un lot d'images")
    test_batch_shape_extraction()
