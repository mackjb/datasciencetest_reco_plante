#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'analyse de l'importance des caractéristiques pour la classification des espèces de plantes.
Utilise différentes stratégies de sélection de caractéristiques pour comparer leur performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import multiprocessing

# Configurer OpenCV pour utiliser tous les cœurs disponibles
cv2.setNumThreads(multiprocessing.cpu_count())

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from joblib import Memory, Parallel, delayed
import time

# Pour la visualisation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif pour la sauvegarde des figures
import seaborn as sns

# Pour les métriques d'évaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Pour l'explication du modèle
import shap

# Import des fonctions existantes
from src.helpers.helpers import (
    PROJECT_ROOT, compute_hu_features, compute_fourier_energy,
    compute_hog_features, compute_pixel_ratio_and_segments, is_image_valid,
    is_black_image, compute_color_statistics, compute_texture_features,
    compute_sharpness_and_contours, compute_additional_fft_features,
    compute_additional_hog_features, compute_shape_features
)

# -----------------------------
# 1. Configuration & Constants
# -----------------------------
# Chemins
DATA_DIR = PROJECT_ROOT / "dataset" / "plantvillage" / "data"

# Listes des colonnes de caractéristiques pour PlantVillage segmented
HANDCRAFTED_FEATURE_COLS = [
    # Moments de Hu
    'phi1_distingue_large_vs_etroit',
    'phi2_distinction_elongation_forme',
    'phi3_asymetrie_maladie',
    'phi4_symetrie_diagonale_forme',
    'phi5_concavite_extremites',
    'phi6_decalage_torsion_maladie',
    'phi7_asymetrie_complexe',
    # Caractéristiques FFT
    'energie_basse_forme_feuille',
    'energie_moyenne_texture_veines',
    'energie_haute_details_maladie',
    'fft_entropy',  # Nouvelle caractéristique
    # Caractéristiques HOG
    'hog_moyenne_contours_forme',
    'hog_ecarttype_texture',
    'hog_entropy',  # Nouvelle caractéristique
    # Caractéristiques de pixel
    'pixel_ratio',
    'leaf_segments',
    # Statistiques de couleur RGB
    'mean_R', 'mean_G', 'mean_B',
    'std_R', 'std_G', 'std_B',
    # Statistiques de couleur HSV
    'mean_H', 'mean_S', 'mean_V',
    # Caractéristiques de texture
    'contrast', 'energy', 'homogeneity', 'dissimilarite', 'correlation',
    # Netteté et contours
    'nettete', 'contour_density',
]

SHAPE_DESCRIPTOR_FEATURE_COLS = [
    'area',
    'perimeter',
    'circularity',
    'solidity',
    'extent',
    'eccentricity',
    'major_axis_length',
    'minor_axis_length',
    'compactness',
    'fractal_dimension',
]

FEATURE_COLS = HANDCRAFTED_FEATURE_COLS + SHAPE_DESCRIPTOR_FEATURE_COLS
CSV_FILE = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all.csv"
TARGET_COLUMN = "species"  # Cible: espèce de la plante
PATH_COLUMN = "filepath"   # Chemin de l'image

# Colonnes de caractéristiques
FEATURE_COLUMNS = FEATURE_COLS  # Toutes les caractéristiques extraites des images

# Autres paramètres
RANDOM_STATE = 42
CACHE_DIR = PROJECT_ROOT / "cache"  # Dossier pour la mise en cache des calculs
CACHE_DIR.mkdir(exist_ok=True)

# Configuration du cache pour joblib.Memory
memory = Memory(CACHE_DIR, verbose=0)

# -----------------------------
# 2. Chargement, Rééchantillonnage & Split
# -----------------------------

def process_and_save_clean_data_with_features(csv_path=CSV_FILE, output_path=None):
    """
    Charge les données propres originales, calcule toutes les features et les sauvegarde dans un CSV.
    
    Args:
        csv_path: Chemin vers le fichier CSV original avec les données propres
        output_path: Chemin où sauvegarder le nouveau CSV avec les features
    
    Returns:
        DataFrame contenant les données avec les features calculées
    """
    print(f"\nTraitement des données originales depuis {csv_path}...")
    # Charger les données originales
    df_original = pd.read_csv(csv_path)
    
    # Préparer les features
    print("Calcul des features pour les données originales...")
    X, y = prepare_features(df_original)
    
    # Créer un DataFrame avec les features
    features_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    features_df[TARGET_COLUMN] = y
    
    # Ajouter les autres colonnes du DataFrame original si nécessaires
    for col in df_original.columns:
        if col != TARGET_COLUMN and col not in FEATURE_COLUMNS:
            features_df[col] = df_original.reset_index(drop=True)[col]
    
    # Sauvegarder le DataFrame
    if output_path:
        # Créer le répertoire si nécessaire
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Sauvegarder au format CSV
        features_df.to_csv(output_path, index=False)
        print(f"Données avec features sauvegardées dans: {output_path}")
    
    return features_df
def augment_class(class_df, num_to_generate, path_col=PATH_COLUMN, data_dir=DATA_DIR, n_jobs=-1):
    """
    Génère de nouveaux exemples pour une classe spécifique en utilisant les transformations.
    
    Args:
        class_df: DataFrame contenant les exemples d'une classe
        num_to_generate: Nombre d'exemples à générer
        path_col: Colonne contenant le chemin vers l'image
        data_dir: Répertoire racine des données
        
    Returns:
        DataFrame contenant les nouveaux exemples générés
    """
    print(f"  - Génération de {num_to_generate} nouveaux exemples...")
    
    # Pour ajouter du bruit gaussien
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.mean = mean
            self.std = std
        
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    
    # Pour la translation (décalage)
    class RandomTranslation(object):
        def __init__(self, max_shift=10):
            self.max_shift = max_shift
            
        def __call__(self, img):
            width, height = img.size
            shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
            shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
            
            try:
                # Essayer avec interpolation (nouvelle API)
                return transforms.functional.affine(
                    img,
                    angle=0,
                    translate=[shift_x, shift_y],
                    scale=1.0,
                    shear=0,
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            except (TypeError, AttributeError):
                try:
                    # Essayer sans spécifier le mode d'interpolation
                    return transforms.functional.affine(
                        img,
                        angle=0,
                        translate=[shift_x, shift_y],
                        scale=1.0,
                        shear=0
                    )
                except Exception as e:
                    print(f"Fallback pour translation: {e}")
                    return img  # Retourner l'image originale si tout échoue
            
        def __repr__(self):
            return self.__class__.__name__ + f'(max_shift={self.max_shift})'
    
    # Pour le zoom in/out
    class RandomZoom(object):
        def __init__(self, min_factor=0.8, max_factor=1.2):
            self.min_factor = min_factor
            self.max_factor = max_factor
            
        def __call__(self, img):
            scale_factor = np.random.uniform(self.min_factor, self.max_factor)
            
            return transforms.functional.affine(
                img,
                angle=0,
                translate=[0, 0],
                scale=scale_factor,
                shear=0
            )
            
        def __repr__(self):
            return self.__class__.__name__ + f'(min_factor={self.min_factor}, max_factor={self.max_factor})'
    
    # Composition de toutes les transformations
    augment = transforms.Compose([
        # Flip (retournement horizontal et vertical)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        
        # Rotation aléatoire (-30°, +30°)
        transforms.RandomRotation(30),
        
        # Crop aléatoire (recadrage)
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        
        # Translation (décalage)
        RandomTranslation(max_shift=20),
        
        # Zoom in/out
        RandomZoom(min_factor=0.85, max_factor=1.15),
        
        # Modification de luminosité, contraste, saturation
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3, 
            saturation=0.3, 
            hue=0.1
        ),
        
        # Conversion en tenseur pour l'ajout de bruit
        transforms.ToTensor(),
        
        # Ajout de bruit gaussien
        AddGaussianNoise(mean=0, std=0.05),
        
        # Gaussian blur et sharpness
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Ajustement de la netteté (sharpness)
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        
        # Reconversion en PIL Image
        transforms.ToPILImage()
    ])
    
    # Fonction pour augmenter une image
    def augment_single_image(idx, row):
        if idx >= num_to_generate:
            return None
        
        img_path = row[path_col]
        try:
            img = Image.open(img_path)
            
            # Appliquer les transformations
            img_aug = augment(img)
            
            # Créer une nouvelle ligne avec des données similaires à la ligne originale
            new_row = row.copy()
            new_row['filename'] = f"augmented_{idx}_{row['filename']}"
            new_row['hash'] = f"augmented_{idx}_{row['hash']}"
            
            return new_row
        except Exception as e:
            print(f"Erreur lors de l'augmentation de {img_path}: {e}")
            return None
    
    # Préparer les entrées pour le traitement parallèle
    start_time = time.time()
    inputs = []
    idx = 0
    
    # Créer la liste d'entrées pour le traitement parallèle
    while len(inputs) < num_to_generate:
        for _, row in class_df.iterrows():
            if len(inputs) >= num_to_generate:
                break
            inputs.append((idx, row))
            idx += 1
    
    # Limiter aux entrées requises
    inputs = inputs[:num_to_generate]
    
    print(f"  - Traitement parallèle de {len(inputs)} augmentations avec {n_jobs} processus...")
    
    # Traitement parallèle
    results = Parallel(n_jobs=n_jobs)(
        delayed(augment_single_image)(i, r) for i, r in inputs
    )
    
    # Filtrer les résultats None
    augmented_rows = [r for r in results if r is not None]
    
    end_time = time.time()
    print(f"  - Augmentation terminée en {end_time - start_time:.2f} secondes")
    
    # Concaténer tous les exemples générés
    if not augmented_rows:
        return pd.DataFrame(columns=class_df.columns)  # DataFrame vide avec mêmes colonnes
        
    return pd.concat(augmented_rows, ignore_index=True)

@memory.cache
def balance_dataset(df, target_col=TARGET_COLUMN, target_counts=None, path_col=PATH_COLUMN, data_dir=DATA_DIR, random_state=RANDOM_STATE):
    """
    Rééquilibre le dataset complet avant le split train/test.
    
    Args:
        df: DataFrame d'origine
        target_col: Colonne cible contenant les classes
        target_counts: Dictionnaire {classe: nombre cible d'exemples}
                      Si None, toutes les classes auront le même nombre défini par une stratégie équilibrée
        path_col: Colonne contenant le chemin vers l'image
        data_dir: Répertoire racine des données
        random_state: État aléatoire pour la reproductibilité
    
    Returns:
        DataFrame rééquilibré
    """
    print("\nRééquilibrage des classes du dataset...")
    
    # 1. Analyser la distribution actuelle
    current_counts = df[target_col].value_counts()
    print("\nDistribution initiale des classes:")
    for cls, count in current_counts.items():
        print(f"  - {cls}: {count} exemples")
    
    # 2. Déterminer les cibles (si non spécifiées)
    if target_counts is None:
        # Stratégie: moyenne pondérée entre médiane et maximum
        # Permet de sur-échantillonner les classes minoritaires sans trop sous-échantillonner les majoritaires
        median_count = current_counts.median()
        max_count = current_counts.max()
        # Pondération: 70% médiane, 30% maximum pour un équilibrage modéré
        target_size = int(0.7 * median_count + 0.3 * max_count)
        target_counts = {cls: target_size for cls in current_counts.index}
    
    print("\nObjectifs de rééquilibrage:")
    for cls, target in sorted(target_counts.items()):
        current = current_counts.get(cls, 0)
        diff = target - current
        if diff > 0:
            print(f"  - {cls}: {current} → {target} (+{diff}, oversampling)")
        elif diff < 0:
            print(f"  - {cls}: {current} → {target} ({diff}, undersampling)")
        else:
            print(f"  - {cls}: {current} (pas de changement)")
    
    # 3. Initialiser la liste pour les lignes du DataFrame rééquilibré
    balanced_rows = []
    
    # 4. Traiter chaque classe
    for cls in current_counts.index:
        subset = df[df[target_col] == cls]
        current = len(subset)
        target = target_counts[cls]
        
        if current > target:  # Undersampling
            # Réduction légère des classes sur-représentées
            print(f"\nClasse {cls}: undersampling de {current} → {target} exemples")
            subset = subset.sample(n=target, random_state=random_state)
            balanced_rows.append(subset)
            
        elif current < target:  # Oversampling
            # Garder tous les exemples originaux
            print(f"\nClasse {cls}: oversampling de {current} → {target} exemples")
            balanced_rows.append(subset)
            
            # Calculer combien d'exemples augmentés sont nécessaires
            to_generate = target - current
            
            # Utiliser les transformations d'augmentation pour générer de nouveaux exemples en parallèle
            augmented = augment_class(subset, to_generate, path_col, data_dir, n_jobs=-1)
            balanced_rows.append(augmented)
        
        else:  # Pas de changement nécessaire
            print(f"\nClasse {cls}: conservée à {current} exemples")
            balanced_rows.append(subset)
    
    # 5. Combiner tous les sous-ensembles et mélanger
    balanced_df = pd.concat(balanced_rows, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Afficher la distribution finale
    final_counts = balanced_df[target_col].value_counts()
    print("\nDistribution après rééquilibrage:")
    for cls, count in final_counts.items():
        print(f"  - {cls}: {count} exemples")
    
    return balanced_df

@memory.cache
def load_and_split_data(csv_path=CSV_FILE, target_col=TARGET_COLUMN, test_size=0.2, random_state=RANDOM_STATE, apply_balancing=True):
    """
    Charge le DataFrame, effectue un rééquilibrage des classes, puis une division stratifiée.
    Mise en cache pour éviter de recharger les données à chaque exécution.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        target_col: Colonne cible pour la stratification
        test_size: Proportion du jeu de test
        random_state: Graine aléatoire pour la reproductibilité
        apply_balancing: Si True, applique le rééquilibrage des classes
        
    Returns:
        tuple: (df_train, df_test) DataFrames d'entraînement et de test
    """
    print(f"\nChargement des données depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Note: Les chemins dans le CSV sont maintenant relatifs et n'ont plus besoin d'être corrigés

    
    # Vérifier que la colonne cible existe
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans le DataFrame. "
                        f"Colonnes disponibles: {', '.join(df.columns)}")
    
    # Vérifier s'il y a des valeurs NaN dans les données
    if df.isna().any().any():
        print("Attention: Le DataFrame contient des valeurs NaN. Nettoyage en cours...")
        print(f"Nombre de lignes avant nettoyage: {len(df)}")
        print("Colonnes avec NaN:", df.isna().sum()[df.isna().sum() > 0])
        df = df.dropna(subset=[target_col])  # Supprimer les lignes avec des NaN dans la cible
        print(f"Nombre de lignes après nettoyage: {len(df)}")
    
    # Rééquilibrer le dataset si demandé
    if apply_balancing:
        df = balance_dataset(df, target_col)
    
    # Vérifier à nouveau pour des NaN après le rééquilibrage
    if df.isna().any().any():
        print("Attention: Des valeurs NaN ont été introduites lors du rééquilibrage. Nettoyage en cours...")
        print("Colonnes avec NaN:", df.isna().sum()[df.isna().sum() > 0])
        # Remplacer les NaN par des valeurs appropriées selon le type de colonne
        # Pour les colonnes numériques, on utilise la médiane
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        # Pour les colonnes catégorielles (y compris la cible), on utilise le mode (valeur la plus fréquente)
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Division stratifiée
    print(f"\nDivision stratifiée en train ({1-test_size:.0%}) / test ({test_size:.0%})...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    # Vérification finale avant split
    if df[target_col].isna().any():
        raise ValueError("La colonne cible contient toujours des NaN après nettoyage!")
    
    train_idx, test_idx = next(sss.split(df, df[target_col]))
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)
    
    print(f"Train set: {len(df_train)} exemples")
    print(f"Test set: {len(df_test)} exemples")
    
    return df_train, df_test

# -----------------------------
# 3. Data Augmentation for Minority
# -----------------------------

def normalize_image(image, method='minmax'):
    """
    Normalise les pixels d'une image avec différentes méthodes.
    
    Args:
        image: Image à normaliser (np.ndarray)
        method: Méthode de normalisation ('minmax', 'zscore', 'robust')
        
    Returns:
        Image normalisée
    """
    # Vérifier que l'image n'est pas vide
    if image is None or image.size == 0:
        return image
    
    if method == 'minmax':
        # Normalisation min-max dans [0,1]
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image
    elif method == 'zscore':
        # Normalisation Z-score
        img_mean = np.mean(image)
        img_std = np.std(image)
        if img_std > 0:
            return (image - img_mean) / img_std
        return image
    elif method == 'robust':
        # Normalisation robuste (utilisant les percentiles pour éviter l'impact des outliers)
        p_low, p_high = np.percentile(image, [2, 98])
        if p_high > p_low:
            return np.clip((image - p_low) / (p_high - p_low), 0, 1)
        return image
    else:
        return image  # Retourner l'image originale si méthode non reconnue
@memory.cache
def extract_features_from_image(img_input, normalize_pixels=True, norm_method='robust', debug=True):
    """
    Extrait toutes les caractéristiques d'une image en utilisant les fonctions existantes.
    
    Args:
        img_input: Chemin vers l'image ou objet PIL.Image
        normalize_pixels: Si True, normalise les pixels de l'image
        norm_method: Méthode de normalisation ('minmax', 'zscore', 'robust')
        
    Returns:
        dict: Dictionnaire des caractéristiques
    """
    try:
        # Détecter si l'entrée est un chemin ou une image PIL
        if isinstance(img_input, (str, Path)):
            # C'est un chemin, charger avec OpenCV
            img_cv = cv2.imread(str(img_input))
            if img_cv is None:
                print(f"Impossible de charger l'image: {img_input}")
                return None
        else:
            # C'est un objet PIL.Image, convertir en format OpenCV
            img_pil = img_input
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Vérifications de base
        # is_image_valid s'attend à un chemin, ne pas l'appeler sur img_cv
        # Nous avons déjà vérifié que l'image n'était pas None avant
        if is_black_image(img_cv):
            return None
            
        # Normaliser l'image si demandé
        if normalize_pixels:
            # La normalisation retourne des valeurs entre 0 et 1 en float64
            img_cv = normalize_image(img_cv, method=norm_method)
            # Reconvertir en uint8 pour éviter les problèmes avec les fonctions d'extraction
            img_cv = (img_cv * 255).astype(np.uint8)
        
        # Convertir en PIL pour compatibilité avec les fonctions existantes
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Extraire les caractéristiques en utilisant les fonctions existantes
        features = {}
        features.update(compute_hu_features(img))
        features.update(compute_fourier_energy(img))
        features.update(compute_hog_features(img))
        features.update(compute_pixel_ratio_and_segments(img))
        
        # Extraire les nouvelles caractéristiques
        features.update(compute_color_statistics(img))
        features.update(compute_texture_features(img))
        features.update(compute_sharpness_and_contours(img))
        features.update(compute_additional_fft_features(img))
        features.update(compute_additional_hog_features(img))
        features.update(compute_shape_features(img))  # Ajout des descripteurs de forme
        
        return features
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques de {img_input}: {e}")
        return None

def augment_minority(df_train, data_dir=DATA_DIR, path_col=PATH_COLUMN, target_col=TARGET_COLUMN):
    """
    Augmente la classe minoritaire pour équilibrer le train set.
    Utilise les transformations de torchvision et recalcule les caractéristiques.
    
    Args:
        df_train: DataFrame d'entraînement
        data_dir: Répertoire racine des données
        path_col: Nom de la colonne contenant le chemin de l'image
        target_col: Nom de la colonne cible
        
    Returns:
        DataFrame étendu avec les mêmes colonnes + caractéristiques recalculées
    """
    print("Augmentation des classes minoritaires...")
    
    # Définir les transformations pour l'augmentation
    # Pour ajouter du bruit gaussien
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.mean = mean
            self.std = std
        
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    
    # Pour la translation (décalage)
    class RandomTranslation(object):
        def __init__(self, max_shift=10):
            self.max_shift = max_shift
            
        def __call__(self, img):
            width, height = img.size
            shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
            shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
            
            try:
                # Essayer avec interpolation (nouvelle API)
                return transforms.functional.affine(
                    img,
                    angle=0,
                    translate=[shift_x, shift_y],
                    scale=1.0,
                    shear=0,
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            except (TypeError, AttributeError):
                try:
                    # Essayer sans spécifier le mode d'interpolation
                    return transforms.functional.affine(
                        img,
                        angle=0,
                        translate=[shift_x, shift_y],
                        scale=1.0,
                        shear=0
                    )
                except Exception as e:
                    print(f"Fallback pour translation: {e}")
                    return img  # Retourner l'image originale si tout échoue
            
        def __repr__(self):
            return self.__class__.__name__ + f'(max_shift={self.max_shift})'
    
    # Pour le zoom in/out
    class RandomZoom(object):
        def __init__(self, min_factor=0.8, max_factor=1.2):
            self.min_factor = min_factor
            self.max_factor = max_factor
            
        def __call__(self, img):
            scale_factor = np.random.uniform(self.min_factor, self.max_factor)
            
            return transforms.functional.affine(
                img,
                angle=0,
                translate=[0, 0],
                scale=scale_factor,
                shear=0
            )
            
        def __repr__(self):
            return self.__class__.__name__ + f'(min_factor={self.min_factor}, max_factor={self.max_factor})'
    
    # Composition de toutes les transformations
    augment = transforms.Compose([
        # Flip (retournement horizontal et vertical)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        
        # Rotation aléatoire (-30°, +30°)
        transforms.RandomRotation(30),
        
        # Crop aléatoire (recadrage)
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        
        # Translation (décalage)
        RandomTranslation(max_shift=20),
        
        # Zoom in/out
        RandomZoom(min_factor=0.85, max_factor=1.15),
        
        # Modification de luminosité, contraste, saturation
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3, 
            saturation=0.3, 
            hue=0.1
        ),
        
        # Conversion en tenseur pour l'ajout de bruit
        transforms.ToTensor(),
        
        # Ajout de bruit gaussien
        AddGaussianNoise(mean=0, std=0.05),
        
        # Gaussian blur et sharpness
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Ajustement de la netteté (sharpness)
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        
        # Reconversion en PIL Image
        transforms.ToPILImage()
    ])
    
    # Compter les occurrences par classe
    counts = df_train[target_col].value_counts()
    max_count = counts.max()
    augmented_rows = []
    
    # Ajouter toutes les lignes originales
    augmented_rows.append(df_train)
    
    # Pour chaque classe minoritaire
    for cls, cnt in counts.items():
        subset = df_train[df_train[target_col] == cls]
        
        if cnt < max_count:
            to_generate = max_count - cnt
            print(f"Classe {cls}: génération de {to_generate} exemples supplémentaires...")
            
            # Nombre de répétitions nécessaires
            reps = int(np.ceil(to_generate / cnt))
            num_generated = 0
            
            for _ in range(reps):
                if num_generated >= to_generate:
                    break
                    
                for _, row in subset.iterrows():
                    if num_generated >= to_generate:
                        break
                        
                    img_path = row[path_col]
                    try:
                        img = Image.open(img_path)
                        # Vérifier le mode de l'image
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img_aug = augment(img)
                        
                        # Extraire les caractéristiques de l'image augmentée
                        feats = extract_features_from_image(img)
                        if feats is None:
                            continue
                            
                        # Créer une nouvelle ligne
                        new_row = row.copy()
                        for k, v in feats.items():
                            new_row[k] = v
                            
                        # Ajouter au DataFrame
                        augmented_rows.append(pd.DataFrame([new_row]))
                        num_generated += 1
                        
                    except Exception as e:
                        print(f"Erreur lors de l'augmentation de {img_path}: {e}")
    
    # Concaténer toutes les lignes
    df_aug = pd.concat(augmented_rows, ignore_index=True)
    print(f"Dataset après augmentation: {len(df_aug)} exemples")
    
    # Vérifier la distribution des classes après augmentation
    new_counts = df_aug[target_col].value_counts()
    for cls, count in new_counts.items():
        print(f"  - {cls}: {count} exemples")
        
    return df_aug

# -----------------------------
# 4. Préparation X/y avec parallélisation
# -----------------------------
@memory.cache
def prepare_features(df, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN, n_jobs=-1):
    """
    Prépare les matrices X et y à partir du DataFrame.
    Utilise parallélisation pour extraire/vérifier les caractéristiques manquantes.
    
    Args:
        df: DataFrame avec les métadonnées et caractéristiques
        feature_cols: Liste des colonnes de caractéristiques
        target_col: Nom de la colonne cible
        n_jobs: Nombre de jobs parallèles (-1 pour utiliser tous les cœurs)
        
    Returns:
        X, y: Matrices de caractéristiques et vecteur cible
    """
    print("Préparation des caractéristiques...")
    
    # Vérifier les colonnes manquantes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
        print("Extraction des caractéristiques manquantes...")
        
        # Fonction pour extraire les caractéristiques d'une ligne
        def extract_features_for_row(row):
            img_path = row[PATH_COLUMN]
            img_path_resolved = None
            img_path_candidates = []
            
            # 1. Essayer le chemin direct tel qu'il est dans le dataframe
            if Path(img_path).exists():
                img_path_resolved = img_path
            else:
                # 2. Essayer avec PROJECT_ROOT si c'est un chemin relatif
                if not Path(img_path).is_absolute():
                    # Si le chemin commence déjà par 'dataset', on suppose qu'il est relatif à PROJECT_ROOT
                    if img_path.startswith('dataset'):
                        candidate_path = PROJECT_ROOT / img_path
                        img_path_candidates.append(candidate_path)
                    # Sinon, on ajoute simplement PROJECT_ROOT
                    else:
                        candidate_path = PROJECT_ROOT / img_path
                        img_path_candidates.append(candidate_path)
                
                # 3. Essayer d'extraire juste le nom de fichier et chercher dans le répertoire des données
                filename = Path(img_path).name
                for root, dirs, files in os.walk(DATA_DIR):
                    if filename in files:
                        candidate_path = Path(root) / filename
                        img_path_candidates.append(candidate_path)
                        break
                
                # Vérifier si un des chemins candidats existe
                for candidate_path in img_path_candidates:
                    if candidate_path.exists():
                        img_path_resolved = candidate_path
                        break
            
            # Si aucun chemin n'est valide, on abandonne
            if img_path_resolved is None:
                # Afficher discrètement pour éviter de polluer la sortie
                return None
                
            try:
                features = extract_features_from_image(img_path_resolved)
                if features is None:
                    # Échec silencieux
                    return None
                return features
            except Exception as e:
                # Exception silencieuse pour éviter de polluer la sortie
                return None
        
        # Extraction parallèle des caractéristiques
        start_time = time.time()
        
        # Réduire la taille du dataframe pour accélérer l'exécution et éviter les erreurs
        print("Utilisation d'un échantillon réduit pour l'analyse d'importance des features...")
        # Échantillonner maximum 500 lignes par classe pour l'analyse
        sample_size = min(500, len(df) // len(df[target_col].unique()))
        df_sample = df.groupby(target_col).apply(lambda x: x.sample(min(sample_size, len(x)))).reset_index(drop=True)
        print(f"Taille de l'échantillon: {len(df_sample)} (réduit de {len(df)})")
        
        # Extraction parallèle des caractéristiques sans utiliser le cache joblib
        print("Désactivation temporaire du cache pour l'extraction parallèle...")
        # Méthode d'extraction: soit parallèle soit séquentielle selon la stabilité
        try:
            # Limiter à 4 jobs maximum pour éviter la surcharge
            actual_n_jobs = min(4, multiprocessing.cpu_count() if n_jobs == -1 else n_jobs)
            print(f"Utilisation de {actual_n_jobs} processus pour l'extraction...")
            
            features_list = [None] * len(df_sample)
            batch_size = 100  # Traiter par lots de 100 pour éviter la surcharge mémoire
            
            for i in range(0, len(df_sample), batch_size):
                end_idx = min(i + batch_size, len(df_sample))
                print(f"Traitement du lot {i//batch_size + 1}/{(len(df_sample) + batch_size - 1)//batch_size}...")
                
                batch_features = Parallel(n_jobs=actual_n_jobs, timeout=300)(
                    delayed(extract_features_for_row)(row) for _, row in df_sample.iloc[i:end_idx].iterrows()
                )
                
                for j, features in enumerate(batch_features):
                    features_list[i + j] = features
        except Exception as e:
            print(f"ERREUR en parallèle: {e}\nBascule en mode séquentiel (plus lent)...")
            # En cas d'échec, on passe en mode séquentiel avec seulement les 500 premières lignes
            df_sample = df_sample.head(500)  # Limiter davantage pour le mode séquentiel
            print(f"Utilisation de seulement {len(df_sample)} exemples en mode séquentiel...")
            features_list = [
                extract_features_for_row(row) for _, row in df_sample.iterrows()
            ]
        
        # Remplacer df par df_sample pour la suite du traitement
        df = df_sample.copy()
        
        end_time = time.time()
        print(f"Extraction terminée en {end_time - start_time:.2f} secondes")
        
        # Analyser les résultats de l'extraction
        success_count = sum(1 for f in features_list if f is not None)
        total_count = len(features_list)
        print(f"\nEXTRACTION: {success_count}/{total_count} images traitées avec succès ({success_count/total_count*100:.1f}%)")
        
        # Si très peu de succès, montrer les 5 premières lignes du dataframe pour diagnostic
        if success_count < total_count * 0.1:
            print("\nAPERÇU DU DATAFRAME (5 premières lignes):")
            print(df.head().to_string())
            
        # Mettre à jour le DataFrame avec les caractéristiques extraites
        success_cols = set()
        for i, features in enumerate(features_list):
            if features is not None:
                # Mettre à jour le DataFrame
                for col, val in features.items():
                    df.loc[i, col] = val
                    success_cols.add(col)
        
        print(f"\nCOLONNES EXTRAITES AVEC SUCCÈS: {len(success_cols)}/{len(feature_cols)}")
        if success_cols:
            print(f"Exemples: {list(success_cols)[:5]}...")
        else:
            print("AUCUNE COLONNE EXTRAITE AVEC SUCCÈS!")
    
    # Vérifier à nouveau les colonnes manquantes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Attention: Certaines colonnes sont toujours manquantes après extraction: {missing_cols}")
        print("Initialisation de ces colonnes avec des valeurs par défaut (0)...")
        
        # Créer les colonnes manquantes avec des valeurs par défaut (0)
        for col in missing_cols:
            df[col] = 0
    
    # Gérer les éventuelles valeurs NaN dans les caractéristiques
    for col in feature_cols:
        if df[col].isna().any():
            print(f"Remplacement des valeurs NaN dans la colonne {col}")
            # Utiliser la médiane pour les valeurs numériques
            df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
    
    # Extraire X et y
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y

# -----------------------------
# 5. Base Classifier & Scaler
# -----------------------------
def create_base_classifier(random_state=RANDOM_STATE):
    """Crée le classifieur de base"""
    return RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=random_state
    )

# -----------------------------
# 6. Define Feature Selectors
# -----------------------------
def create_selectors(random_state=RANDOM_STATE):
    """Crée les différents sélecteurs de caractéristiques"""
    return {
        # Univariate: F-test, keep top 10 features
        'univariate_f': SelectKBest(score_func=f_classif, k=10),
        
        # RFE avec RandomForest
        'rfe_rf': RFE(
            estimator=RandomForestClassifier(n_estimators=100, random_state=random_state),
            n_features_to_select=10
        ),
        
        # Model-based: L1 LogReg
        'sfm_l1': SelectFromModel(
            estimator=LogisticRegression(
                penalty='l1', solver='liblinear', random_state=random_state
            ),
            threshold='median'
        ),
        
        # Model-based: Tree
        'sfm_tree': SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=random_state),
            threshold='median'
        )
    }

# -----------------------------
# 7. Evaluation Function
# -----------------------------
def evaluate_selectors(X, y, selectors, classifier, scaler=RobustScaler(), cv=5, random_state=RANDOM_STATE):
    """
    Compare plusieurs stratégies de sélection via CV.
    
    Args:
        X: Matrice de caractéristiques
        y: Vecteur cible
        selectors: Dictionnaire de sélecteurs de caractéristiques
        classifier: Classifieur de base
        scaler: Scaler pour normaliser les données
        cv: Nombre de folds pour la validation croisée
        random_state: État aléatoire pour la reproductibilité
        
    Returns:
        dict: Résultats des différentes stratégies
    """
    results = {}
    feature_importances = {}
    selected_features_idx = {}
    
    # Pour chaque sélecteur
    for name, selector in selectors.items():
        print(f"\nÉvaluation du sélecteur: {name}")
        
        # Créer le pipeline
        pipe = Pipeline([
            ('scaler', scaler),
            ('feat_sel', selector),
            ('clf', classifier)
        ])
        
        # Création d'une validation croisée stratifiée explicite
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

         # Évaluation par validation croisée stratifiée
        cv_results = cross_validate(
            pipe, X, y, cv=stratified_cv,  # Utilisation explicite de StratifiedKFold
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            return_train_score=True,
            return_estimator=True  # Pour récupérer les modèles entraînés
        )
        
        # Stocker les résultats moyens
        results[name] = {
            'accuracy': np.mean(cv_results['test_accuracy']),
            'f1_macro': np.mean(cv_results['test_f1_macro']),
            'precision_macro': np.mean(cv_results['test_precision_macro']),
            'recall_macro': np.mean(cv_results['test_recall_macro']),
            'train_accuracy': np.mean(cv_results['train_accuracy']),
            'train_f1_macro': np.mean(cv_results['train_f1_macro']),
        }
        
        # Récupérer les caractéristiques sélectionnées pour chaque fold
        selected_features = []
        importances = []
        
        for estimator in cv_results['estimator']:
            # Récupérer l'index des caractéristiques sélectionnées
            if name == 'univariate_f':
                mask = estimator.named_steps['feat_sel'].get_support()
            elif name == 'rfe_rf':
                mask = estimator.named_steps['feat_sel'].get_support()
            elif name in ['sfm_l1', 'sfm_tree']:
                mask = estimator.named_steps['feat_sel'].get_support()
            else:
                mask = np.ones(X.shape[1], dtype=bool)
                
            selected_features.append(mask)
            
            # Récupérer les importances des caractéristiques du classifieur final
            if hasattr(estimator.named_steps['clf'], 'feature_importances_'):
                # Pour les modèles basés sur les arbres (RandomForest)
                # Ces importances sont calculées sur les caractéristiques sélectionnées
                fold_importances = estimator.named_steps['clf'].feature_importances_
                
                # Créer un vecteur d'importances de la taille originale
                full_importances = np.zeros(X.shape[1])
                full_importances[mask] = fold_importances
                importances.append(full_importances)
        
        # Calculer la fréquence de sélection pour chaque caractéristique
        selection_frequency = np.mean(selected_features, axis=0)
        selected_features_idx[name] = selection_frequency
        
        # Calculer l'importance moyenne des caractéristiques si disponible
        if importances:
            feature_importances[name] = np.mean(importances, axis=0)
        
    return results, feature_importances, selected_features_idx

# -----------------------------
# 8. Affichage des résultats
# -----------------------------
def print_results(results, feature_importances, selected_features_idx, feature_cols):
    """
    Affiche les résultats de l'évaluation et l'importance des caractéristiques.
    
    Args:
        results: Résultats des différentes stratégies
        feature_importances: Importances des caractéristiques pour chaque stratégie
        selected_features_idx: Index des caractéristiques sélectionnées
        feature_cols: Noms des colonnes de caractéristiques
    """
    print("\n" + "="*50)
    print("RÉSULTATS DE LA SÉLECTION DE CARACTÉRISTIQUES")
    print("="*50)
    
    # Afficher les métriques de performance
    print("\nMétriques de performance:")
    for name, scores in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Test - Accuracy: {scores['accuracy']:.3f}")
        print(f"  Test - F1 macro: {scores['f1_macro']:.3f}")
        print(f"  Test - Precision macro: {scores['precision_macro']:.3f}")
        print(f"  Test - Recall macro: {scores['recall_macro']:.3f}")
        print(f"  Train - Accuracy: {scores['train_accuracy']:.3f}")
        print(f"  Train - F1 macro: {scores['train_f1_macro']:.3f}")
    
    # Afficher les caractéristiques les plus importantes pour chaque stratégie
    print("\nCaractéristiques importantes par stratégie:")
    for name, freq in selected_features_idx.items():
        print(f"\n{name.upper()} - Fréquence de sélection:")
        
        # Trier par fréquence de sélection
        indices = np.argsort(-freq)
        for i in indices[:10]:  # Top 10
            if freq[i] > 0:
                print(f"  {feature_cols[i]}: {freq[i]:.2f}")
    
    # Si les importances sont disponibles (pour les méthodes basées sur les arbres)
    print("\nImportance des caractéristiques (si disponible):")
    for name, importances in feature_importances.items():
        if importances is not None and np.any(importances):
            print(f"\n{name.upper()} - Feature Importance:")
            
            # Trier par importance
            indices = np.argsort(-importances)
            for i in indices[:10]:  # Top 10
                if importances[i] > 0:
                    print(f"  {feature_cols[i]}: {importances[i]:.4f}")

# -----------------------------
# 8.1 Visualisation des résultats
# -----------------------------
def plot_feature_importance(feature_importances, selected_features_idx, feature_cols):
    """
    Génère et sauvegarde des graphiques en barres pour visualiser les caractéristiques 
    les plus importantes pour chaque stratégie.
    
    Args:
        feature_importances: Importances des caractéristiques pour chaque stratégie
        selected_features_idx: Index des caractéristiques sélectionnées
        feature_cols: Noms des colonnes de caractéristiques
    """
    print("\nGénération des graphiques d'importance des caractéristiques...")
    
    # Créer un répertoire pour les figures si nécessaire
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    # Pour chaque stratégie, créer un graphique
    for strategy_type, data_dict in [
        ("frequency", selected_features_idx),
        ("importance", feature_importances)
    ]:
        for name, values in data_dict.items():
            # Vérifier que les valeurs existent
            if values is None or not np.any(values):
                continue
                
            # Trier les valeurs et sélectionner les 10 premières
            indices = np.argsort(-values)
            top_indices = indices[:10]
            top_values = values[top_indices]
            
            # Ne garder que les valeurs positives
            pos_mask = top_values > 0
            if not np.any(pos_mask):
                continue
                
            top_indices = top_indices[pos_mask]
            top_values = top_values[pos_mask]
            top_names = [feature_cols[i] for i in top_indices]
            
            # Créer le graphique
            plt.figure(figsize=(12, 8))
            
            # Créer des barres horizontales
            y_pos = np.arange(len(top_names))
            plt.barh(y_pos, top_values, align='center')
            
            # Ajouter les noms des caractéristiques et les étiquettes
            plt.yticks(y_pos, top_names)
            
            # Ajouter un titre et des légendes
            if strategy_type == "frequency":
                plt.title(f'Top 10 Caractéristiques - {name} (Fréquence de Sélection)', fontsize=16)
                plt.xlabel('Fréquence de Sélection')
            else:
                plt.title(f'Top 10 Caractéristiques - {name} (Importance)', fontsize=16)
                plt.xlabel('Importance')
                
            plt.ylabel('Caractéristiques')
            plt.tight_layout()
            
            # Sauvegarder le graphique
            fig_path = figures_dir / f"{name}_{strategy_type}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Graphique sauvegardé: {fig_path}")

# -----------------------------
# 8.2 Explication SHAP
# -----------------------------
def explain_with_shap(X, y, feature_cols, random_state=RANDOM_STATE):
    """
    Utilise SHAP pour expliquer les prédictions du modèle RandomForest.
    
    Args:
        X: Matrice de caractéristiques
        y: Vecteur cible
        feature_cols: Noms des colonnes de caractéristiques
        random_state: État aléatoire pour la reproductibilité
    """
    try:
        print("\n" + "="*50)
        print("ANALYSE D'EXPLICABILITÉ AVEC SHAP")
        print("="*50)
        
        # Limiter la taille du jeu de données pour des raisons de performances
        max_samples = 2000
        if len(y) > max_samples:
            indices = np.random.RandomState(random_state).choice(
                len(y), max_samples, replace=False
            )
            X_sample = X[indices]
            y_sample = y[indices]
            print(f"Utilisation d'un sous-échantillon de {len(y_sample)} exemples pour SHAP")
        else:
            X_sample = X
            y_sample = y
            
        # Créer le répertoire pour les figures SHAP
        figures_dir = PROJECT_ROOT / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Entraîner un modèle simple pour l'explication
        print("Entraînement du modèle pour l'explication SHAP...")
        model = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=random_state
        )
        model.fit(X_sample, y_sample)
        
        # Utiliser une approche simplifiée pour calculer l'importance des features
        print("Calcul de l'importance des caractéristiques...")
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        # Créer un graphique d'importance des caractéristiques
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx[:15])), feature_importance[sorted_idx[:15]])
        plt.yticks(range(len(sorted_idx[:15])), [feature_cols[i] for i in sorted_idx[:15]])
        plt.title("Importance des caractéristiques selon RandomForest")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance_rf.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Tentative de calcul des valeurs SHAP...")
        try:
            # Essayer le calcul SHAP sur un sous-échantillon encore plus petit si nécessaire
            small_sample_size = min(500, len(X_sample))
            sub_indices = np.random.RandomState(random_state).choice(
                len(X_sample), small_sample_size, replace=False
            )
            X_small = X_sample[sub_indices]
            
            # Créer l'explainer et calculer les valeurs SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_small)
            
            # Vérifier si les dimensions correspondent
            if isinstance(shap_values, list):
                print(f"Forme des valeurs SHAP (liste de {len(shap_values)} éléments)")
                for i, sv in enumerate(shap_values):
                    print(f"  Classe {i}: {sv.shape}")
                
                # Créer un DataFrame pour faciliter la visualisation
                X_df = pd.DataFrame(X_small, columns=feature_cols)
                
                # Générer des graphiques pour quelques classes
                for i in range(min(3, len(shap_values))):
                    class_label = np.unique(y_sample)[i]
                    # Calculer l'importance absolue moyenne pour chaque feature
                    importance = np.abs(shap_values[i]).mean(0)
                    idx = np.argsort(importance)[::-1][:10]  # Top 10 features
                    
                    plt.figure(figsize=(10, 6))
                    plt.barh(range(len(idx)), importance[idx])
                    plt.yticks(range(len(idx)), [feature_cols[j] for j in idx])
                    plt.title(f"SHAP Feature Importance - Classe {class_label}")
                    plt.tight_layout()
                    plt.savefig(figures_dir / f"shap_importance_class_{class_label}.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    
                # Créer une heatmap SHAP par classe et feature
                print("Génération de la heatmap SHAP par classe et feature...")
                try:
                    # Obtenir les noms de classes
                    class_names = [str(c) for c in np.unique(y_sample)]
                    
                    # Calculer l'importance absolue moyenne de chaque feature pour chaque classe
                    feature_importance_by_class = np.zeros((len(feature_cols), len(class_names)))
                    
                    # Sélectionner les top features pour la visualisation
                    top_n_features = min(20, len(feature_cols))  # Limiter à 20 features max pour la lisibilité
                    all_importance = np.array([np.abs(sv).mean(0) for sv in shap_values])
                    global_importance = all_importance.mean(0)
                    top_feature_idx = np.argsort(global_importance)[::-1][:top_n_features]
                    top_feature_names = [feature_cols[i] for i in top_feature_idx]
                    
                    # Remplir la matrice d'importance
                    for class_idx in range(len(class_names)):
                        for feat_idx, feat_orig_idx in enumerate(top_feature_idx):
                            feature_importance_by_class[feat_idx, class_idx] = np.abs(shap_values[class_idx][:, feat_orig_idx]).mean()
                    
                    # Créer la heatmap
                    plt.figure(figsize=(14, 10))
                    plt.title("Importance des caractéristiques par classe (SHAP)", fontsize=16)
                    
                    # Créer le heatmap avec seaborn pour une meilleure visualisation
                    ax = sns.heatmap(
                        feature_importance_by_class, 
                        annot=True, 
                        fmt=".3f",
                        cmap="viridis", 
                        xticklabels=class_names, 
                        yticklabels=top_feature_names,
                        cbar_kws={'label': 'Importance SHAP (absolue moyenne)'}
                    )
                    
                    plt.xlabel("Classes", fontsize=12)
                    plt.ylabel("Caractéristiques", fontsize=12)
                    plt.tight_layout()
                    
                    # Sauvegarder la heatmap
                    plt.savefig(figures_dir / "shap_heatmap_class_feature.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print("Heatmap SHAP sauvegardée : figures/shap_heatmap_class_feature.png")
                except Exception as e:
                    print(f"Erreur lors de la création de la heatmap SHAP: {e}")
                    traceback.print_exc()
            else:
                print(f"Forme des valeurs SHAP: {shap_values.shape}")
                # Format spécifique (n_samples, n_features, n_classes)
                if len(shap_values.shape) == 3:
                    print("Génération de la heatmap SHAP par classe et feature pour le format 3D...")
                    try:
                        # Obtenir les noms de classes
                        class_names = [str(c) for c in np.unique(y_sample)]
                        n_samples, n_features, n_classes = shap_values.shape
                        
                        # Sélectionner les top features pour la visualisation
                        top_n_features = min(20, n_features)  # Limiter à 20 features max pour la lisibilité
                        
                        # Calculer l'importance absolue moyenne de chaque feature pour chaque classe
                        # Pour le format 3D, nous prenons la moyenne sur tous les échantillons
                        feature_importance_by_class = np.zeros((top_n_features, n_classes))
                        feature_importance_global = np.abs(shap_values).mean(axis=0).mean(axis=1)  # Moyenne sur échantillons et classes
                        top_feature_idx = np.argsort(feature_importance_global)[::-1][:top_n_features]
                        top_feature_names = [feature_cols[i] for i in top_feature_idx]
                        
                        # Calcul des valeurs d'importance pour la heatmap
                        for class_idx in range(n_classes):
                            for i, feat_idx in enumerate(top_feature_idx):
                                # Moyenne absolue pour chaque feature et classe
                                feature_importance_by_class[i, class_idx] = np.abs(shap_values[:, feat_idx, class_idx]).mean()
                        
                        # Créer la heatmap
                        plt.figure(figsize=(14, 10))
                        plt.title("Importance des caractéristiques par classe (SHAP)", fontsize=16)
                        
                        # Créer la heatmap avec seaborn
                        ax = sns.heatmap(
                            feature_importance_by_class, 
                            annot=True, 
                            fmt=".3f",
                            cmap="viridis", 
                            xticklabels=class_names, 
                            yticklabels=top_feature_names,
                            cbar_kws={'label': 'Importance SHAP (absolue moyenne)'}
                        )
                        
                        plt.xlabel("Classes", fontsize=12)
                        plt.ylabel("Caractéristiques", fontsize=12)
                        plt.tight_layout()
                        
                        # Sauvegarder la heatmap
                        plt.savefig(figures_dir / "shap_heatmap_class_feature.png", dpi=300, bbox_inches='tight')
                        plt.close()
                        print("Heatmap SHAP sauvegardée : figures/shap_heatmap_class_feature.png")
                    except Exception as e:
                        print(f"Erreur lors de la création de la heatmap SHAP 3D: {e}")
                        traceback.print_exc()
                else:
                    print("Format inattendu pour les valeurs SHAP - graphique non généré")
        except Exception as e:
            print(f"Erreur lors du calcul SHAP: {e}")
            print("Passage à l'importance des caractéristiques basique")
        
        print("SHAP analysis terminée.")
    except Exception as e:
        print(f"Erreur lors de l'analyse SHAP: {e}")
        import traceback
        traceback.print_exc()
    
    # Fin de la fonction

# -----------------------------
# 8.3 Heatmaps pour la sélection de caractéristiques
# -----------------------------
def plot_feature_selection_heatmaps(feature_importances, selected_features_idx, feature_cols, output_dir):
    """
    Crée des heatmaps pour visualiser les résultats de la sélection de caractéristiques
    avec différentes méthodes (UNIVARIATE_F, RFE_RF, SFM_L1, SFM_TREE).
    
    Args:
        feature_importances: Dictionnaire des importances des caractéristiques pour chaque stratégie
        selected_features_idx: Dictionnaire des fréquences de sélection pour chaque stratégie
        feature_cols: Liste des noms des caractéristiques
        output_dir: Répertoire où sauvegarder les figures
    """
    print("Génération des heatmaps pour la sélection de caractéristiques...")
    
    try:
        # 1. Sélectionner les caractéristiques les plus importantes globalement
        all_importances = []
        
        # Récupérer les importances pour chaque stratégie
        for name, data in feature_importances.items():
            all_importances.append(data)
        
        # Calculer l'importance globale moyenne
        if all_importances:
            global_importance = np.mean(all_importances, axis=0)
        else:
            # Si pas d'importances disponibles, utiliser les fréquences
            all_frequencies = []
            for name, freq in selected_features_idx.items():
                all_frequencies.append(freq)
            
            global_importance = np.mean(all_frequencies, axis=0) if all_frequencies else np.ones(len(feature_cols))
        
        # Sélectionner les top caractéristiques pour la visualisation
        top_n = min(25, len(feature_cols))  # Limiter à 25 caractéristiques pour la lisibilité
        top_indices = np.argsort(-global_importance)[:top_n]
        top_features = [feature_cols[i] for i in top_indices]
        
        # 2. Créer la matrice pour la heatmap
        selector_names = list(feature_importances.keys())
        
        # Créer deux heatmaps: une pour la fréquence, une pour l'importance
        
        # 2.1 Heatmap pour les fréquences de sélection
        # Initialiser la matrice
        heatmap_freq = np.zeros((len(top_features), len(selector_names)))
        
        # Remplir la matrice avec les fréquences
        for j, name in enumerate(selector_names):
            freq_data = selected_features_idx[name]
            for i, feat_idx in enumerate(top_indices):
                heatmap_freq[i, j] = freq_data[feat_idx]
        
        # Créer la heatmap de fréquence
        plt.figure(figsize=(14, 10))
        plt.title("Heatmap des Fréquences de Sélection des Caractéristiques", fontsize=16)
        
        # Créer le heatmap
        ax = sns.heatmap(
            heatmap_freq, 
            annot=True, 
            fmt=".2f",
            cmap="viridis", 
            xticklabels=selector_names, 
            yticklabels=top_features,
            cbar_kws={'label': 'Fréquence de Sélection'}
        )
        
        plt.xlabel("Sélecteurs", fontsize=12)
        plt.ylabel("Caractéristiques", fontsize=12)
        plt.tight_layout()
        
        # Sauvegarder la heatmap
        filename = "heatmap_frequency_by_selector.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap de fréquence sauvegardée : {filename}")
        
        # 2.2 Heatmap pour les importances
        # Initialiser la matrice
        heatmap_imp = np.zeros((len(top_features), len(selector_names)))
        
        # Remplir la matrice avec les importances
        for j, name in enumerate(selector_names):
            imp_data = feature_importances[name]
            for i, feat_idx in enumerate(top_indices):
                heatmap_imp[i, j] = imp_data[feat_idx]
        
        # Créer la heatmap d'importance
        plt.figure(figsize=(14, 10))
        plt.title("Heatmap des Importances des Caractéristiques", fontsize=16)
        
        # Créer le heatmap
        ax = sns.heatmap(
            heatmap_imp, 
            annot=True, 
            fmt=".3f",
            cmap="viridis", 
            xticklabels=selector_names, 
            yticklabels=top_features,
            cbar_kws={'label': 'Importance'}
        )
        
        plt.xlabel("Sélecteurs", fontsize=12)
        plt.ylabel("Caractéristiques", fontsize=12)
        plt.tight_layout()
        
        # Sauvegarder la heatmap
        filename = "heatmap_importance_by_selector.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap d'importance sauvegardée : {filename}")
        
        # 3. Créer une heatmap comparative qui montre toutes les méthodes côte à côte avec un score normalisé
        plt.figure(figsize=(16, 12))
        plt.title("Comparaison des Méthodes de Sélection de Caractéristiques", fontsize=16)
        
        # Calculer un score normalisé pour chaque caractéristique et sélecteur
        combined_data = np.zeros((len(top_features), len(selector_names)))
        
        # Pour chaque sélecteur, normaliser les scores
        for j, name in enumerate(selector_names):
            # Obtenir les fréquences
            freq = selected_features_idx[name]
            # Obtenir les importances
            imp = feature_importances[name]
            
            for i, feat_idx in enumerate(top_indices):
                # Score de base = fréquence
                score = freq[feat_idx]
                
                # Incorporer l'importance
                if np.max(imp) > 0:  # Éviter la division par zéro
                    # Score = fréquence * (1 + importance normalisée)
                    imp_norm = imp[feat_idx] / np.max(imp)
                    score = score * (1 + imp_norm) / 2
                
                combined_data[i, j] = score
        
        # Créer la heatmap comparative
        ax = sns.heatmap(
            combined_data,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=selector_names,
            yticklabels=top_features,
            cbar_kws={'label': 'Score Combiné (Fréquence et Importance)'}
        )
        
        plt.xlabel("Sélecteurs", fontsize=12)
        plt.ylabel("Caractéristiques", fontsize=12)
        plt.tight_layout()
        
        filename = "heatmap_comparaison_methodes.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap comparative sauvegardée : {filename}")
        
    except Exception as e:
        print(f"Erreur lors de la création des heatmaps de sélection de caractéristiques: {e}")
        import traceback
        traceback.print_exc()

# -----------------------------
# 9. Main Function
# -----------------------------
def main():
    """
    Fonction principale qui exécute l'ensemble du processus.
    """
    print("\n" + "="*50)
    print("ANALYSE DE L'IMPORTANCE DES CARACTÉRISTIQUES")
    print("="*50)
    
    # Créer le répertoire results s'il n'existe pas
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # 0. Traiter et sauvegarder les données originales avec features
    clean_output_path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_with_features_data_plantvillage_segmented_all.csv"
    process_and_save_clean_data_with_features(CSV_FILE, clean_output_path)
    
    # 1. Charger les données avec rééquilibrage avant le split train/test
    df_train, df_test = load_and_split_data(apply_balancing=True)
    
    # Note: Plus besoin d'utiliser augment_minority car le rééquilibrage est fait avant le split
    df_train_aug = df_train
    
    # 3. Préparer X/y pour train et test
    X_train, y_train = prepare_features(df_train_aug)
    X_test, y_test = prepare_features(df_test)
    
    # Sauvegarder les données augmentées avec features sous CSV
    print("\nSauvegarde des features après oversampling...")
    # Créer un nouveau DataFrame avec les caractéristiques extraites
    features_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    # Ajouter la colonne cible (espèce)
    features_df[TARGET_COLUMN] = y_train
    # Ajouter les autres colonnes du DataFrame original si nécessaires
    for col in df_train_aug.columns:
        if col != TARGET_COLUMN and col not in FEATURE_COLUMNS:
            features_df[col] = df_train_aug.reset_index(drop=True)[col]
    
    # Créer le répertoire si nécessaire
    csv_output_path = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "oversampling_with_features_data_plantvillage_segmented_all.csv"
    csv_output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Sauvegarder au format CSV
    features_df.to_csv(csv_output_path, index=False)
    print(f"Données sauvegardées dans: {csv_output_path}")
    
    # Combiner pour l'évaluation complète
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    
    # 4. Créer les objets nécessaires
    base_clf = create_base_classifier()
    scaler = RobustScaler()
    selectors = create_selectors()
    
    # 5. Évaluer les sélecteurs
    results, feature_importances, selected_features_idx = evaluate_selectors(
        X_all, y_all, selectors, base_clf, scaler
    )
    
    # 6. Afficher les résultats textuels
    print_results(results, feature_importances, selected_features_idx, FEATURE_COLUMNS)
    
    # 7. Générer les visualisations
    plot_feature_importance(feature_importances, selected_features_idx, FEATURE_COLUMNS)
    
    # Ajouter des heatmaps pour la sélection de caractéristiques
    plot_feature_selection_heatmaps(feature_importances, selected_features_idx, FEATURE_COLUMNS, figures_dir)
    
    # 8. Sélectionner les meilleures caractéristiques
    print("\n" + "="*50)
    print("SÉLECTION DES MEILLEURES CARACTÉRISTIQUES")
    print("="*50)
    
    selected_features, feature_ranking = select_best_features(
        results, feature_importances, selected_features_idx, FEATURE_COLUMNS, threshold=0.5
    )
    
    # Afficher les caractéristiques sélectionnées
    print(f"Nombre de caractéristiques sélectionnées: {len(selected_features)}")
    print("\nTop 15 des caractéristiques les plus importantes:")
    print(selected_features[['feature', 'selection_frequency', 'final_score']].head(15).to_string(index=False))
    
    # 9. Visualiser les caractéristiques sélectionnées
    plot_selected_features(feature_ranking, output_dir=figures_dir)
    
    # 10. Entraîner un modèle final avec les caractéristiques sélectionnées
    print("\n" + "="*50)
    print("MODÈLE FINAL AVEC CARACTÉRISTIQUES SÉLECTIONNÉES")
    print("="*50)
    
    final_model, final_scaler = train_final_model(
        X_all, y_all, selected_features['feature'].tolist(), FEATURE_COLUMNS
    )
    
    # Sauvegarder les résultats
    feature_ranking.to_csv(results_dir / "feature_ranking.csv", index=False)
    
    # 11. Explication SHAP pour une compréhension plus fine
    explain_with_shap(X_all, y_all, FEATURE_COLUMNS)
    
    return results, feature_importances, selected_features_idx, selected_features

# -----------------------------
# 10. Feature Selection
# -----------------------------
def select_best_features(results, feature_importances, selected_features_idx, feature_names, threshold=0.5):
    """
    Sélectionne les meilleures caractéristiques en combinant les scores des différentes méthodes
    
    Args:
        results: Résultats d'évaluation des sélecteurs
        feature_importances: Importances des caractéristiques par méthode
        selected_features_idx: Indices des caractéristiques sélectionnées par méthode
        feature_names: Noms des caractéristiques
        threshold: Seuil de sélection (fréquence minimale)
        
    Returns:
        Liste des meilleures caractéristiques avec leurs scores combinés
    """
    # Combiner les scores de sélection
    n_features = len(feature_names)
    combined_scores = np.zeros(n_features)
    
    # Calculer un score moyen basé sur toutes les méthodes
    for method, scores in feature_importances.items():
        combined_scores += scores / len(feature_importances)
    
    # Calculer la fréquence de sélection
    selection_frequency = np.zeros(n_features)
    for method, selected in selected_features_idx.items():
        selection_frequency += selected
    selection_frequency /= len(selected_features_idx)
    
    # Combiner score d'importance et fréquence de sélection
    final_scores = (combined_scores + selection_frequency) / 2
    
    # Créer un DataFrame avec les scores
    feature_ranking = pd.DataFrame({
        'feature': feature_names,
        'importance_score': combined_scores,
        'selection_frequency': selection_frequency,
        'final_score': final_scores
    })
    
    # Trier par score final
    feature_ranking = feature_ranking.sort_values('final_score', ascending=False)
    
    # Sélectionner les caractéristiques au-dessus du seuil
    selected_features = feature_ranking[feature_ranking['selection_frequency'] >= threshold]
    
    return selected_features, feature_ranking

def plot_selected_features(feature_ranking, output_dir=None):
    """
    Visualise les caractéristiques sélectionnées avec leurs scores
    
    Args:
        feature_ranking: DataFrame avec les scores des caractéristiques
        output_dir: Dossier de sortie pour sauvegarder le graphique
    """
    plt.figure(figsize=(12, 8))
    
    # Prendre les 20 meilleures caractéristiques
    top_features = feature_ranking.head(20)
    
    # Créer un barplot
    sns.barplot(x='final_score', y='feature', data=top_features)
    
    plt.title('Top 20 des caractéristiques les plus importantes', fontsize=14)
    plt.xlabel('Score combiné (importance + fréquence de sélection)', fontsize=12)
    plt.ylabel('Caractéristique', fontsize=12)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / "selected_features.png", dpi=300, bbox_inches='tight')
    
    plt.close()

def train_final_model(X, y, selected_features, feature_names, random_state=RANDOM_STATE):
    """
    Entraîne un modèle final avec les caractéristiques sélectionnées
    
    Args:
        X: Matrice de caractéristiques
        y: Vecteur cible
        selected_features: Liste des caractéristiques sélectionnées
        feature_names: Noms de toutes les caractéristiques
        random_state: État aléatoire
        
    Returns:
        Modèle final entraîné et scaler utilisé
    """
    # Obtenir les indices des caractéristiques sélectionnées
    selected_indices = [feature_names.index(feat) for feat in selected_features]
    
    # Sous-ensemble des données avec caractéristiques sélectionnées
    X_selected = X[:, selected_indices]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Normalisation
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraîner le modèle final
    final_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,
        class_weight='balanced',
        random_state=random_state
    )
    
    final_model.fit(X_train_scaled, y_train)
    
    # Évaluer sur le test set
    y_pred = final_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\nModèle final avec {len(selected_features)} caractéristiques:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return final_model, scaler

# -----------------------------
# 11. SHAP Explainer
# -----------------------------
if __name__ == '__main__':
    try:
        start_time = time.time()
        results, feature_importances, selected_features_idx, selected_features = main()
        end_time = time.time()
        print(f"\nTemps total d'exécution: {end_time - start_time:.2f} secondes")
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        print("\nLe script a rencontré une erreur. Veuillez vérifier les messages ci-dessus.")
        sys.exit(1)
