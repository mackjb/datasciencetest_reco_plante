#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'analyse de l'importance des caractéristiques pour la classification des espèces de plantes.
Utilise différentes stratégies de sélection de caractéristiques pour comparer leur performance.
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, train_test_split
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
    PROJECT_ROOT, 
    compute_hu_features, 
    compute_fourier_energy, 
    compute_hog_features, 
    compute_pixel_ratio_and_segments,
    compute_color_statistics,
    compute_texture_features,
    compute_sharpness_and_contours,
    compute_additional_fft_features,
    compute_additional_hog_features,
    is_image_valid,
    is_black_image
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
def augment_class(class_df, num_to_generate, path_col=PATH_COLUMN, data_dir=DATA_DIR):
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
    
    # Générer de nouveaux exemples
    augmented_rows = []
    num_generated = 0
    
    # Boucle jusqu'à atteindre le nombre d'exemples demandé
    while num_generated < num_to_generate:
        # Parcourir le DataFrame de la classe
        for _, row in class_df.iterrows():
            if num_generated >= num_to_generate:
                break
                
            img_path = row[path_col]
            try:
                img = Image.open(img_path)
                
                # Vérifier le mode de l'image
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_aug = augment(img)
                
                # Extraire les caractéristiques de l'image augmentée
                feats = extract_features_from_image(img_aug)
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
            
            # Utiliser les transformations d'augmentation pour générer de nouveaux exemples
            augmented = augment_class(subset, to_generate, path_col, data_dir)
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
        random_state: État aléatoire pour la reproductibilité
        apply_balancing: Si True, applique le rééquilibrage avant le split
        
    Returns:
        df_train, df_test: DataFrames d'entraînement et de test
    """
    print(f"Chargement des données depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Vérifier que la colonne cible existe
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans le DataFrame. "
                        f"Colonnes disponibles: {', '.join(df.columns)}")
    
    # Rééquilibrer le dataset si demandé
    if apply_balancing:
        df = balance_dataset(df, target_col)
    
    # Division stratifiée
    print(f"\nDivision stratifiée en train ({1-test_size:.0%}) / test ({test_size:.0%})...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
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
def extract_features_from_image(img_input, normalize_pixels=True, norm_method='robust'):
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
        if not is_image_valid(img_cv) or is_black_image(img_cv):
            return None
            
        # Normaliser l'image si demandé
        if normalize_pixels:
            img_cv = normalize_image(img_cv, method=norm_method)
        
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
        
        return features
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques de {img_path}: {e}")
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
            features = extract_features_from_image(img_path)
            return features
        
        # Extraction parallèle des caractéristiques
        start_time = time.time()
        features_list = Parallel(n_jobs=n_jobs)(
            delayed(extract_features_for_row)(row) for _, row in df.iterrows()
        )
        end_time = time.time()
        print(f"Extraction terminée en {end_time - start_time:.2f} secondes")
        
        # Mettre à jour le DataFrame avec les caractéristiques extraites
        for i, features in enumerate(features_list):
            if features is not None:
                for col, val in features.items():
                    df.loc[i, col] = val
    
    # Vérifier à nouveau les colonnes manquantes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Impossible d'extraire toutes les caractéristiques requises: {missing_cols}")
    
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
        
        # Évaluation par validation croisée
        cv_results = cross_validate(
            pipe, X, y, cv=cv,
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
    print("\n" + "="*50)
    print("ANALYSE D'EXPLICABILITÉ AVEC SHAP")
    print("="*50)
    
    # Créer un répertoire pour les figures SHAP si nécessaire
    shap_dir = Path("figures/shap")
    shap_dir.mkdir(exist_ok=True, parents=True)
    
    # Sous-échantillonner les données pour accélérer le calcul de SHAP
    # (SHAP peut être intensif en calcul pour de grands ensembles de données)
    X_sample, _, y_sample, _ = train_test_split(
        X, y, test_size=0.8, random_state=random_state, stratify=y
    )
    print(f"Utilisation d'un sous-échantillon de {len(X_sample)} exemples pour SHAP")
    
    # Normaliser les données avec RobustScaler
    scaler = RobustScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)
    
    # Entraîner un RandomForest sur les données
    print("Entraînement du modèle RandomForest pour l'explication SHAP...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_state, 
        class_weight='balanced'
    )
    rf.fit(X_sample_scaled, y_sample)
    
    # Créer un explainer SHAP
    print("Calcul des valeurs SHAP...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample_scaled)
    
    # Récupérer les noms des classes
    class_names = np.unique(y)
    num_classes = len(class_names)
    
    print(f"Génération des visualisations SHAP pour {num_classes} classes...")
    
    # 1. Summary plot global (toutes les caractéristiques, toutes les classes combinées)
    print("Création du summary plot global...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X_sample_scaled, 
        feature_names=feature_cols,
        class_names=[str(c) for c in class_names],
        show=False
    )
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary_all_classes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pour chaque classe, créer un summary plot spécifique
    for i, class_name in enumerate(class_names):
        print(f"Création du summary plot pour la classe {class_name}...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[i], 
            X_sample_scaled,
            feature_names=feature_cols,
            plot_type='bar',
            show=False
        )
        plt.title(f"Impact des caractéristiques sur la classe: {class_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(shap_dir / f"shap_summary_class_{i}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Pour les top 5 features, générer des partial dependence plots simplifiés
    # Nous évitons d'utiliser shap.dependence_plot car il cause une erreur d'indexation
    # dans ce cas d'usage multiclasse
    
    # Identifier les top features par leur importance moyenne absolue
    feature_importance = np.abs(np.array(shap_values)).mean(0).mean(0)
    top_indices = np.argsort(-feature_importance)[:5]
    top_features = [feature_cols[i] for i in top_indices]
    
    # Pour chaque top feature, créer un summary plot spécifique plutôt que des scatter plots
    # Nous évitons les scatter plots en raison de problèmes de compatibilité de taille
    print(f"Création des visualisations pour les top 5 caractéristiques...")
    
    # Créer un beeswarm plot combiné pour les top 5 caractéristiques (toutes classes confondues)
    plt.figure(figsize=(12, 7))
    top_features_idx = np.argsort(-np.abs(np.array(shap_values)).mean(0).mean(0))[:5]
    
    # Fusionner les valeurs SHAP pour toutes les classes pour ces caractéristiques
    vals = np.array(shap_values).mean(0)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        vals, 
        X_sample_scaled, 
        plot_type="bar",
        feature_names=feature_cols,
        max_display=10,  # Afficher uniquement les 10 principales caractéristiques
        show=False
    )
    plt.title("Impact global moyen des caractéristiques (toutes classes)", fontsize=14)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_top_features_global.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Créer un plot pour chacune des 3 premières classes
    for class_idx in range(min(3, len(class_names))):
        plt.figure(figsize=(10, 6))
        class_name = class_names[class_idx]
        
        # Générer un bar plot pour cette classe spécifique
        shap.summary_plot(
            shap_values[class_idx],
            X_sample_scaled,
            plot_type="bar",
            feature_names=feature_cols,
            max_display=10,  # Afficher uniquement les 10 principales caractéristiques
            show=False
        )
        
        plt.title(f"Impact des caractéristiques pour la classe: {class_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(shap_dir / f"shap_top_features_class_{class_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()
            
    # Fin des plots pour les classes individuelles
    
    # Créer un heatmap des valeurs SHAP absolues moyennes pour les top features
    plt.figure(figsize=(14, 8))
    
    # Préparer les données pour le heatmap (top features x classes)
    heatmap_data = np.zeros((len(top_features), len(class_names)))
    for i, idx in enumerate(top_indices):
        for j in range(len(class_names)):
            heatmap_data[i, j] = np.abs(shap_values[j][:, idx]).mean()
    
    # Créer le heatmap
    ax = plt.gca()
    im = ax.imshow(heatmap_data, cmap='viridis')
    
    # Ajouter les étiquettes
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_xticklabels([str(name) for name in class_names], rotation=45, ha='right')
    ax.set_yticklabels(top_features)
    
    # Ajouter les valeurs dans le heatmap
    for i in range(len(top_features)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                       ha="center", va="center", color="white" if heatmap_data[i, j] > heatmap_data.mean() else "black")
    
    plt.colorbar(im, ax=ax)
    plt.title("Impact moyen des caractéristiques principales par classe", fontsize=14)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_feature_impact_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Les visualisations SHAP ont été sauvegardées dans {shap_dir}")

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
    
    # 1. Charger les données avec rééquilibrage avant le split train/test
    df_train, df_test = load_and_split_data(apply_balancing=True)
    
    # Note: Plus besoin d'utiliser augment_minority car le rééquilibrage est fait avant le split
    df_train_aug = df_train
    
    # 3. Préparer X/y pour train et test
    X_train, y_train = prepare_features(df_train_aug)
    X_test, y_test = prepare_features(df_test)
    
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
    start_time = time.time()
    results, feature_importances, selected_features_idx, selected_features = main()
    end_time = time.time()
    print(f"\nTemps total d'exécution: {end_time - start_time:.2f} secondes")
