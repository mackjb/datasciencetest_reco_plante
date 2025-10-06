#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour vérifier l'extraction des caractéristiques sur une seule image
"""

import sys
import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

# Importer les fonctions nécessaires de my_feature_importance.py
from my_feature_importance import (
    extract_features_from_image, 
    PROJECT_ROOT, 
    DATA_DIR, 
    CSV_FILE
)

def test_extraction_single_image():
    """Test l'extraction des caractéristiques sur une seule image"""
    # Charger le DataFrame
    print(f"Chargement du CSV depuis {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # Prendre la première image
    first_row = df.iloc[0]
    img_path = first_row['filepath']
    print(f"Image à tester: {img_path}")
    
    # Résoudre le chemin complet
    if not Path(img_path).is_absolute():
        if img_path.startswith('dataset'):
            img_path = PROJECT_ROOT / img_path
        else:
            img_path = PROJECT_ROOT / img_path
    
    print(f"Chemin résolu: {img_path}")
    print(f"Le fichier existe: {Path(img_path).exists()}")
    
    # Créer notre propre version de débogage de extract_features_from_image
    print("\nExtraction des caractéristiques avec débogage...")
    try:
        # Étape 1: Charger l'image avec OpenCV
        print("1. Chargement de l'image avec OpenCV...")
        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            print(f"ERREUR: Impossible de charger l'image: {img_path}")
            return
        print(f"   Image chargée: {img_cv.shape}")
        
        # Étape 2: Vérifier la validité de l'image
        print("2. Vérification de la validité de l'image...")
        from src.helpers.helpers import is_black_image
        # is_image_valid attend un chemin, pas un tableau NumPy
        # Comme nous avons vérifié que l'image n'était pas None à l'étape précédente,
        # nous savons qu'elle est valide
        print("   Image valide: Oui (img_cv n'est pas None)")
        
        if is_black_image(img_cv):
            print("ERREUR: Image entièrement noire")
            return
        print("   Image non-noire: Oui")
        
        # Étape 3: Normaliser l'image
        print("3. Normalisation de l'image...")
        from my_feature_importance import normalize_image
        try:
            # La normalisation retourne des valeurs entre 0 et 1 en float64
            img_cv = normalize_image(img_cv, method='robust')
            # Reconvertir en uint8 pour éviter les problèmes avec OpenCV
            img_cv = (img_cv * 255).astype(np.uint8)
            print("   Normalisation réussie avec conversion en uint8")
        except Exception as e:
            print(f"ERREUR: Normalisation échouée: {e}")
            return
        
        # Étape 4: Convertir en PIL
        print("4. Conversion en image PIL...")
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        print(f"   Image PIL créée: {img.size}, {img.mode}")
        
        # Étape 5: Tester chaque fonction d'extraction individuellement
        print("\n5. Test des fonctions d'extraction individuellement:")
        from src.helpers.helpers import (
            compute_hu_features, 
            compute_fourier_energy, 
            compute_hog_features, 
            compute_pixel_ratio_and_segments,
            compute_color_statistics,
            compute_texture_features,
            compute_sharpness_and_contours,
            compute_additional_fft_features,
            compute_additional_hog_features
        )
        
        # Créer un dictionnaire pour stocker les résultats
        features = {}
        
        try:
            print("\n5.1 compute_hu_features...")
            hu_features = compute_hu_features(img)
            print(f"   Résultat: {hu_features}")
            features.update(hu_features)
        except Exception as e:
            print(f"ERREUR: compute_hu_features a échoué: {e}")
        
        try:
            print("\n5.2 compute_fourier_energy...")
            fourier_features = compute_fourier_energy(img)
            print(f"   Résultat: {fourier_features}")
            features.update(fourier_features)
        except Exception as e:
            print(f"ERREUR: compute_fourier_energy a échoué: {e}")
        
        try:
            print("\n5.3 compute_hog_features...")
            hog_features = compute_hog_features(img)
            print(f"   Résultat: {hog_features}")
            features.update(hog_features)
        except Exception as e:
            print(f"ERREUR: compute_hog_features a échoué: {e}")
        
        try:
            print("\n5.4 compute_pixel_ratio_and_segments...")
            pixel_features = compute_pixel_ratio_and_segments(img)
            print(f"   Résultat: {pixel_features}")
            features.update(pixel_features)
        except Exception as e:
            print(f"ERREUR: compute_pixel_ratio_and_segments a échoué: {e}")
        
        try:
            print("\n5.5 compute_color_statistics...")
            color_features = compute_color_statistics(img)
            print(f"   Résultat: {color_features}")
            features.update(color_features)
        except Exception as e:
            print(f"ERREUR: compute_color_statistics a échoué: {e}")
        
        try:
            print("\n5.6 compute_texture_features...")
            texture_features = compute_texture_features(img)
            print(f"   Résultat: {texture_features}")
            features.update(texture_features)
        except Exception as e:
            print(f"ERREUR: compute_texture_features a échoué: {e}")
        
        try:
            print("\n5.7 compute_sharpness_and_contours...")
            sharpness_features = compute_sharpness_and_contours(img)
            print(f"   Résultat: {sharpness_features}")
            features.update(sharpness_features)
        except Exception as e:
            print(f"ERREUR: compute_sharpness_and_contours a échoué: {e}")
        
        try:
            print("\n5.8 compute_additional_fft_features...")
            add_fft_features = compute_additional_fft_features(img)
            print(f"   Résultat: {add_fft_features}")
            features.update(add_fft_features)
        except Exception as e:
            print(f"ERREUR: compute_additional_fft_features a échoué: {e}")
        
        try:
            print("\n5.9 compute_additional_hog_features...")
            add_hog_features = compute_additional_hog_features(img)
            print(f"   Résultat: {add_hog_features}")
            features.update(add_hog_features)
        except Exception as e:
            print(f"ERREUR: compute_additional_hog_features a échoué: {e}")
        
        print(f"\nRE-VÉRIFICATION: {len(features)} caractéristiques extraites:")
        for key, value in features.items():
            print(f"  - {key}: {value}")
        
        return features
    except Exception as e:
        print(f"ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_extraction_single_image()
