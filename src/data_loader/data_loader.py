import glob
import pandas as pd
import numpy as np
from pathlib import Path

from src.helpers.helpers import PROJECT_ROOT, compute_hu_features, compute_fourier_energy, compute_hog_features, compute_pixel_ratio_and_segments, is_image_valid, is_black_image
from PIL import Image
import cv2
import os
import hashlib
import re
from typing import Tuple
from torchvision import transforms

data_root: Path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'data'
default_dataset: str = 'plantvillage dataset/segmented'

# Listes des colonnes de caractéristiques pour PlantVillage segmented
HANDCRAFTED_FEATURE_COLS = [
    'phi1_distingue_large_vs_etroit',
    'phi2_distinction_elongation_forme',
    'phi3_asymetrie_maladie',
    'phi4_symetrie_diagonale_forme',
    'phi5_concavite_extremites',
    'phi6_decalage_torsion_maladie',
    'phi7_asymetrie_complexe',
    'energie_basse_forme_feuille',
    'energie_moyenne_texture_veines',
    'energie_haute_details_maladie',
    'hog_moyenne_contours_forme',
    'hog_ecarttype_texture',
    'pixel_ratio',
    'leaf_segments',
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


def dataset_to_clean_dataframe(subfolder: str = default_dataset) -> pd.DataFrame:
    """
    Charge les images du dataset dans un DataFrame et filtre automatiquement les images non valides et les duplicats.
    
    Args:
        subfolder: Sous-dossier du dataset à traiter
        
    Returns:
        DataFrame contenant les métadonnées des images nettoyées (sans NA ni duplicats)
    """
    folder_path = data_root / subfolder
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Le dossier {folder_path} est introuvable.")

    records = []
    # Parcours des classes (sous-dossiers)
    for class_dir in folder_path.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        # Recherche des images dans la classe
        for pattern in ('*.jpg', '*.jpeg', '*.png'):
            for img_path in class_dir.glob(pattern):
                file_size = img_path.stat().st_size
                filename = img_path.name
                extension = img_path.suffix.lower().lstrip('.')
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        num_channels = len(img.getbands())
                        aspect_ratio = width / height if height else None
                except Exception:
                    width = height = mode = num_channels = aspect_ratio = None

                records.append({
                    'filepath': str(img_path),
                    'filename': filename,
                    'extension': extension,
                    'file_size': file_size,
                    'label': class_name,
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'num_channels': num_channels,
                    'aspect_ratio': aspect_ratio,
                })

    df = pd.DataFrame(records)
    
    # Validité et duplicatas
    print(f"Images totales chargées: {len(df)}")
    df['is_image_valid'] = df['filepath'].apply(is_image_valid)
    df['is_black'] = df['filepath'].apply(is_black_image)
    df['is_na'] = (~df['is_image_valid']) | df['is_black']
    df['hash'] = df['filepath'].apply(lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest())
    df['is_duplicate_after_first'] = df['hash'].duplicated(keep='first')
    
    # Filtrer les images non valides et les duplicats
    df_clean = df[(~df['is_na']) & (~df['is_duplicate_after_first'])].copy()
    df = df.drop(columns=['hash'])
    print(f"Images après filtrage (sans NA ni duplicats): {len(df_clean)}")
    
    # Extraire les colonnes species et disease à partir du label
    df_clean['species'] = df_clean['label'].apply(lambda x: x.split('___')[0] if '___' in x else x)
    df_clean['disease'] = df_clean['label'].apply(lambda x: x.split('___')[1] if '___' in x else 'unknown')
    print("Colonnes 'species' et 'disease' extraites du label.")
    
    # Sauvegarder automatiquement dans le fichier CSV
    csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # Créer le répertoire si nécessaire
    df_clean.to_csv(csv_path, index=False)
    print(f"Données nettoyées sauvegardées dans {csv_path}")
    
    return df_clean


def generate_clean_images(csv_path: Path = None, 
                        output_dir: Path = None, 
                        size: Tuple[int, int] = (256, 256)) -> pd.DataFrame:
    """
    Génère des images nettoyées (redimensionnées à 256x256) à partir du CSV déjà filtré.
    
    Args:
        csv_path: Chemin vers le fichier CSV nettoyé (si None, utilise le chemin par défaut)
        output_dir: Répertoire de sortie pour les images propres (si None, utilise le chemin par défaut)
        size: Taille des images redimensionnées (défaut: 256x256)
        
    Returns:
        DataFrame contenant les métadonnées des images nettoyées et redimensionnées
    """
    # Utiliser les chemins par défaut si non spécifiés
    if csv_path is None:
        csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'clean_images'
    
    # S'assurer que le répertoire de sortie existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger le DataFrame nettoyé
    print(f"Chargement du CSV nettoyé : {csv_path}")
    df_clean = pd.read_csv(csv_path)
    
    # Créer les répertoires pour chaque classe
    for label in df_clean['label'].unique():
        label_dir = output_dir / label
        label_dir.mkdir(exist_ok=True)
    
    # Redimensionner et sauvegarder les images
    print("Redimensionnement et sauvegarde des images propres...")
    rows = []
    for _, row in df_clean.iterrows():
        src_path = Path(row['filepath'])
        if not src_path.exists():
            continue
            
        dst_dir = output_dir / row['label']
        dst_path = dst_dir / f"{src_path.stem}.png"
        
        try:
            with Image.open(src_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
                img_resized.save(dst_path, format="PNG")
                
            # Mettre à jour les informations de l'image
            new_row = row.copy()
            new_row['filepath'] = str(dst_path)
            new_row['filename'] = dst_path.name
            new_row['extension'] = 'png'
            new_row['width'] = size[0]
            new_row['height'] = size[1]
            new_row['file_size'] = os.path.getsize(dst_path)
            rows.append(new_row)
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {src_path}: {e}")
    
    # Créer un nouveau DataFrame avec les images traitées
    df_processed = pd.DataFrame(rows)
    
    # Sauvegarder le DataFrame mis à jour
    clean_csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage.csv'
    clean_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(clean_csv_path, index=False)
    print(f"Données nettoyées sauvegardées dans {clean_csv_path}")
    print(f"Images nettoyées ({len(df_processed)}) sauvegardées dans {output_dir}")
    
    return df_processed


def augment_minority_classes(df_clean: pd.DataFrame = None, 
                            output_dir: Path = None,
                            clean_csv_path: Path = None,
                            image_col: str = 'filepath',
                            label_col: str = 'label',
                            target_size: Tuple[int, int] = (256, 256)) -> pd.DataFrame:
    """
    Augmente les images des classes minoritaires pour équilibrer le dataset.
    
    Args:
        df_clean: DataFrame contenant les métadonnées des images nettoyées (si None, charge depuis clean_csv_path)
        output_dir: Répertoire de sortie pour les images augmentées (si None, utilise le chemin par défaut)
        clean_csv_path: Chemin vers le fichier CSV des images nettoyées (utilisé si df_clean est None)
        image_col: Nom de la colonne contenant les chemins d'images
        label_col: Nom de la colonne contenant les étiquettes des classes
        target_size: Taille cible des images augmentées
        
    Returns:
        DataFrame contenant les métadonnées des images nettoyées et augmentées
    """
    # Utiliser les chemins par défaut si non spécifiés
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'augmented_images'
    
    if df_clean is None:
        if clean_csv_path is None:
            clean_csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage.csv'
        print(f"Chargement du CSV des images nettoyées : {clean_csv_path}")
        df_clean = pd.read_csv(clean_csv_path)
    
    # S'assurer que le répertoire de sortie existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compter le nombre d'images par classe
    counts = df_clean[label_col].value_counts()
    max_count = counts.max()
    print(f"Distribution des classes avant augmentation:\n{counts}")
    print(f"Classe majoritaire contient {max_count} images")
    
    # Créer les répertoires pour chaque classe
    for label in df_clean[label_col].unique():
        label_dir = output_dir / label
        label_dir.mkdir(exist_ok=True)
    
    # Liste pour stocker les nouvelles lignes des images augmentées
    augmented_rows = []
    
    # Pour chaque classe minoritaire, générer des images augmentées
    print("Génération des images augmentées pour les classes minoritaires...")
    for cls, count in counts.items():
        # Calculer combien d'images il faut ajouter
        num_needed = max_count - count
        if num_needed <= 0:
            print(f"Classe {cls} : aucune augmentation nécessaire ({count} images)")
            continue
        
        print(f"Classe {cls} : génération de {num_needed} images supplémentaires")
        
        # Calculer combien d'augmentations par image existante
        subset = df_clean[df_clean[label_col] == cls]
        augs_per_image = (num_needed // count) + (1 if num_needed % count else 0)
        
        # Transformations pour augmenter les images
        transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(target_size, scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        
        # Pour chaque image existante, créer des versions augmentées
        aug_count = 0
        for _, row in subset.iterrows():
            src_path = Path(row[image_col])
            if not src_path.exists():
                continue
                
            try:
                with Image.open(src_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Générer les versions augmentées
                    for i in range(augs_per_image):
                        if aug_count >= num_needed:
                            break
                            
                        # Appliquer les transformations
                        img_tensor = transforms.ToTensor()(img)
                        augmented_tensor = transform(img_tensor)
                        augmented_img = transforms.ToPILImage()(augmented_tensor)
                        
                        # Sauvegarder l'image augmentée
                        dst_dir = output_dir / cls
                        dst_path = dst_dir / f"{src_path.stem}_aug_{i}.png"
                        augmented_img.save(dst_path, format="PNG")
                        
                        # Créer une nouvelle ligne pour l'image augmentée
                        new_row = row.copy()
                        new_row[image_col] = str(dst_path)
                        new_row['filename'] = dst_path.name
                        new_row['extension'] = 'png'
                        new_row['width'] = target_size[0]
                        new_row['height'] = target_size[1]
                        new_row['file_size'] = os.path.getsize(dst_path)
                        new_row['is_augmented'] = True  # Marquer comme augmentée
                        augmented_rows.append(new_row)
                        aug_count += 1
            except Exception as e:
                print(f"Erreur lors de l'augmentation de l'image {src_path}: {e}")
    
    # Ajouter une colonne is_augmented aux données d'origine (False par défaut)
    if 'is_augmented' not in df_clean.columns:
        df_clean['is_augmented'] = False
    
    # Combiner les données d'origine et les données augmentées
    df_augmented = pd.DataFrame(augmented_rows) if augmented_rows else pd.DataFrame(columns=df_clean.columns)
    df_combined = pd.concat([df_clean, df_augmented], ignore_index=True)
    
    # Sauvegarder le DataFrame combiné
    combined_csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'combined_data_plantvillage.csv'
    df_combined.to_csv(combined_csv_path, index=False)
    print(f"Données combinées ({len(df_combined)} images) sauvegardées dans {combined_csv_path}")
    print(f"Nouvelles images augmentées ({len(df_augmented)}) sauvegardées dans {output_dir}")
    
    # Afficher la distribution finale
    final_counts = df_combined[label_col].value_counts()
    print(f"Distribution des classes après augmentation:\n{final_counts}")
    
    return df_combined


def generate_plantvillage_images(force_refresh: bool = False, subfolder: str = default_dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fonction principale pour générer les images nettoyées et augmentées.
    
    Args:
        force_refresh: Si True, régénère toutes les images même si les fichiers CSV existent déjà
        subfolder: Sous-dossier du dataset à traiter
        
    Returns:
        Tuple contenant (DataFrame des images nettoyées, DataFrame des images nettoyées et augmentées)
    """
    # Vérifier si les CSV existent déjà
    clean_csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage.csv'
    combined_csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'combined_data_plantvillage.csv'
    
    if not force_refresh and clean_csv_path.exists() and combined_csv_path.exists():
        print("Chargement des fichiers CSV existants...")
        df_clean = pd.read_csv(clean_csv_path)
        df_combined = pd.read_csv(combined_csv_path)
        print(f"Images nettoyées : {len(df_clean)}")
        print(f"Images totales après augmentation : {len(df_combined)}")
        return df_clean, df_combined
    
    # 0. Générer le DataFrame nettoyé avec la nouvelle fonction
    print("\n=== GÉNÉRATION DU DATASET NETTOYÉ ===")
    df_filtered = dataset_to_clean_dataframe(subfolder=subfolder)
    
    # 1. Générer les images nettoyées (redimensionnées)
    print("\n=== GÉNÉRATION DES IMAGES REDIMENSIONNÉES ===")
    csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    df_clean = generate_clean_images(csv_path=csv_path)
    
    # 2. Générer les images augmentées pour les classes minoritaires
    print("\n=== GÉNÉRATION DES IMAGES AUGMENTÉES ===")
    df_combined = augment_minority_classes(df_clean=df_clean)
    
    return df_clean, df_combined
