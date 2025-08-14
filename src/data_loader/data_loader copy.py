import glob
import pandas as pd
import numpy as np
from pathlib import Path

from src.helpers.helpers import PROJECT_ROOT, compute_hu_features, compute_fourier_energy, compute_hog_features, compute_pixel_ratio_and_segments
from PIL import Image, ImageStat
import cv2
import os
import hashlib
import re
from typing import Tuple
from torchvision import transforms


# Feature column lists for PlantVillage segmented
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


def extract_features_for_df(df, image_col='filepath'):
    """Compute handcrafted and shape descriptor features for each image in df."""
    def _extract(row):
        path = row[image_col]
        try:
            with Image.open(path) as img:
                gray = np.array(img.convert('L'))
        except Exception:
            return pd.Series({col: None for col in FEATURE_COLS})
        hu = compute_hu_features(gray)
        fourier = compute_fourier_energy(gray)
        hog = compute_hog_features(gray)
        pix = compute_pixel_ratio_and_segments(gray)
        shape = compute_shape_descriptors(gray)
        feature_dict = {}
        for col in HANDCRAFTED_FEATURE_COLS:
            if col in hu:
                feature_dict[col] = hu[col]
            elif col in fourier:
                feature_dict[col] = fourier[col]
            elif col in hog:
                feature_dict[col] = hog[col]
            elif col in pix:
                feature_dict[col] = pix[col]
            else:
                feature_dict[col] = None
        for col in SHAPE_DESCRIPTOR_FEATURE_COLS:
            feature_dict[col] = shape.get(col)
        return pd.Series(feature_dict)
    feats_df = df.apply(_extract, axis=1)
    return feats_df

def fractal_dimension(Z):
    """Estimate fractal dimension of a binary contour image Z using box counting."""
    assert Z.ndim == 2
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return np.count_nonzero(S)
    h, w = Z.shape
    min_dim = min(h, w)
    max_exp = int(np.floor(np.log2(min_dim)))
    sizes = 2**np.arange(max_exp, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    sizes = np.array([s for s, c in zip(sizes, counts) if c > 0])
    counts = np.array([c for c in counts if c > 0])
    if len(sizes) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def compute_shape_descriptors(gray):
    """Compute leaf shape descriptor metrics from a grayscale image array."""
    mask = gray > 0
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return dict(
            area=np.nan,
            perimeter=np.nan,
            circularity=np.nan,
            solidity=np.nan,
            extent=np.nan,
            eccentricity=np.nan,
            major_axis_length=np.nan,
            minor_axis_length=np.nan,
            compactness=np.nan,
            fractal_dimension=np.nan
        )
    cnt = max(contours, key=cv2.contourArea)
    area = float(mask.sum())
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = float(w * h)
    extent = area / rect_area if rect_area > 0 else 0.0
    if len(cnt) >= 5:
        (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(cnt)
        major_axis = float(max(axis1, axis2))
        minor_axis = float(min(axis1, axis2))
        eccentricity = major_axis / minor_axis if minor_axis > 0 else 0.0
    else:
        major_axis = 0.0
        minor_axis = 0.0
        eccentricity = 0.0
    compactness = (perimeter**2 / area) if area > 0 else 0.0
    contour_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_mask, [cnt], -1, 1, 1)
    fractal_dim = fractal_dimension(contour_mask)
    return dict(
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        solidity=solidity,
        extent=extent,
        eccentricity=eccentricity,
        major_axis_length=major_axis,
        minor_axis_length=minor_axis,
        compactness=compactness,
        fractal_dimension=fractal_dim
    )


def is_black_image(image_path, threshold=10):
    """
    Returns True if the image is mostly black based on the mean grayscale value.
    """
    with Image.open(image_path).convert("L") as img_gray:
        stat = ImageStat.Stat(img_gray)
        mean_val = stat.mean[0]
    return mean_val < threshold

def is_image_valid(image_path: str) -> bool:
    """Vérifie que l'image n'est pas corrompue."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

# Base path for the PlantVillage dataset directories
# Utilise la constante PROJECT_ROOT pour localiser le dossier data
data_root: Path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'data'

def _load_dataset(subfolder: str) -> pd.DataFrame:
    """
    Charge un sous-dossier du dataset PlantVillage et retourne un DataFrame
    avec les colonnes de base et indicateurs de validité.

    :param subfolder: Nom du dossier à charger (par ex. 'plantvillage dataset')
    :return: pandas.DataFrame
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
    # validité et duplicatas
    df['is_image_valid'] = df['filepath'].apply(is_image_valid)
    df['is_black'] = df['filepath'].apply(is_black_image)
    df['is_na'] = (~df['is_image_valid']) | df['is_black']
    df['hash'] = df['filepath'].apply(lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest())
    df['is_duplicate_after_first'] = df['hash'].duplicated(keep='first')
    df = df.drop(columns=['hash'])
    return df


def load_plantvillage_all() -> pd.DataFrame:
    """
    Charge l'intégralité du dataset PlantVillage.

    :return: pandas.DataFrame avec colonnes 'filepath' et 'label'.
    """
    return _load_dataset('plantvillage dataset/segmented')


def load_plantvillage_five_images() -> pd.DataFrame:
    """
    Charge un sous-ensemble du dataset PlantVillage contenant cinq images par classe.

    :return: pandas.DataFrame avec colonnes 'filepath' et 'label'.
    """
    return _load_dataset('plantvillage_5images/segmented')


def generate_raw_data_plantvillage_segmented_all() -> pd.DataFrame:
    """
    Génère le fichier raw_data_plantvillage_segmented_all.csv dans dataset/plantvillage/csv/
    avec toutes les colonnes de métadonnées (is_na, is_image_valid, is_black, is_duplicate_after_first).
    """
    # Charger toutes les données
    df_raw = load_plantvillage_all()
    
    # Créer le répertoire CSV s'il n'existe pas
    csv_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le CSV raw
    raw_csv = csv_dir / 'raw_data_plantvillage_segmented_all.csv'
    df_raw.to_csv(raw_csv, index=False)
    print(f"Raw data CSV enregistré: {raw_csv}")
    
    return df_raw


def generate_clean_data_plantvillage_segmented_all() -> pd.DataFrame:
    """
    Charge, nettoie, enrichit le dataset PlantVillage segmented et renvoie le DataFrame.
    (Ne sauvegarde plus le CSV.)
    """
    """
    Charge, nettoie, enrichit et sauvegarde le dataset PlantVillage segmented.
    Retourne le DataFrame clean.
    """
    df_raw = load_plantvillage_all()
    # Filtrage des NA et duplicates
    df_clean = df_raw[~(df_raw['is_na'] | df_raw['is_duplicate_after_first'])].drop_duplicates(subset='filepath').copy()

    def extract_features(row):
        path = row['filepath']
        try:
            with Image.open(path) as img:
                gray = np.array(img.convert('L'))
            hu = compute_hu_features(gray)
            fourier = compute_fourier_energy(gray)
            hog = compute_hog_features(gray)
            pix = compute_pixel_ratio_and_segments(gray)
            shape = compute_shape_descriptors(gray)
            return pd.Series({
                'phi1_distingue_large_vs_etroit': hu['phi1_distingue_large_vs_etroit'],
                'phi2_distinction_elongation_forme': hu['phi2_distinction_elongation_forme'],
                'phi3_asymetrie_maladie': hu['phi3_asymetrie_maladie'],
                'phi4_symetrie_diagonale_forme': hu['phi4_symetrie_diagonale_forme'],
                'phi5_concavite_extremites': hu['phi5_concavite_extremites'],
                'phi6_decalage_torsion_maladie': hu['phi6_decalage_torsion_maladie'],
                'phi7_asymetrie_complexe': hu['phi7_asymetrie_complexe'],
                'energie_basse_forme_feuille': fourier['energie_basse_forme_feuille'],
                'energie_moyenne_texture_veines': fourier['energie_moyenne_texture_veines'],
                'energie_haute_details_maladie': fourier['energie_haute_details_maladie'],
                'hog_moyenne_contours_forme': hog['hog_moyenne_contours_forme'],
                'hog_ecarttype_texture': hog['hog_ecarttype_texture'],
                'pixel_ratio': pix['pixel_ratio'],
                                'leaf_segments': pix['leaf_segments'],
                'area': shape['area'],
                'perimeter': shape['perimeter'],
                'circularity': shape['circularity'],
                'solidity': shape['solidity'],
                'extent': shape['extent'],
                'eccentricity': shape['eccentricity'],
                'major_axis_length': shape['major_axis_length'],
                'minor_axis_length': shape['minor_axis_length'],
                'compactness': shape['compactness'],
                'fractal_dimension': shape['fractal_dimension'],
            })
        except Exception:
            # Return None for all features if extraction fails
            return pd.Series({col: None for col in FEATURE_COLS})


    feats_df = extract_features_for_df(df_clean, image_col='filepath')
    df_clean = pd.concat([df_clean, feats_df], axis=1)

    return df_clean


# def augment_minority_classes_cv2(
#     df, image_col='filepath', label_col='species', force_refresh: bool = False
# ) -> pd.DataFrame:


#     # Generation mode: create missing augmented images
#     counts = df[label_col].value_counts()
#     max_count = counts.max()
#     augmented_rows = []
#     for cls, count in counts.items():
#         needing = max_count - count
#         if needing <= 0:
#             continue
#         n_per_img = (needing // count) + (1 if needing % count else 0)
#         subset = df[df[label_col] == cls]
#         species_dir = aug_dir / cls
#         species_dir.mkdir(parents=True, exist_ok=True)
#         for _, row in subset.iterrows():
#             img = cv2.imread(row[image_col])
#             if img is None:
#                 continue
#             h, w = img.shape[:2]
#             base = Path(row[image_col]).stem
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.RandomRotation(20),
#                 transforms.RandomResizedCrop((h, w), scale=(0.9, 1.1)),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2),
#             ])
#             for i in range(n_per_img):
#                 augmented = transform(img_rgb)
#                 save_path = species_dir / f"{base}_aug_{i}.png"
#                 augmented.save(save_path)
#                 new_row = row.copy()
#                 new_row[image_col] = str(save_path)
#                 augmented_rows.append(new_row)
#     return pd.DataFrame(augmented_rows)


#     # mode génération : créer les images manquantes
#     counts = df[label_col].value_counts()
#     max_count = counts.max()
#     augmented_rows = []
#     for cls, count in counts.items():
#         needing = max_count - count
#         if needing <= 0:
#             continue
#         n_per_img = (needing // count) + (1 if needing % count else 0)
#         subset = df[df[label_col] == cls]
#         species_dir = aug_dir / cls
#         species_dir.mkdir(parents=True, exist_ok=True)
#         for _, row in subset.iterrows():
#             img = cv2.imread(row[image_col])
#             if img is None:
#                 continue
#             h, w = img.shape[:2]
#             base = Path(row[image_col]).stem
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.RandomRotation(20),
#                 transforms.RandomResizedCrop((h, w), scale=(0.9, 1.1)),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2),
#             ])
#             for i in range(n_per_img):
#                 augmented = transform(img_rgb)
#                 save_path = species_dir / f"{base}_aug_{i}.png"
#                 augmented.save(save_path)
#                 new_row = row.copy()
#                 new_row[image_col] = str(save_path)
#                 augmented_rows.append(new_row)
#     return pd.DataFrame(augmented_rows)

def generate_clean_and_resized(force_refresh: bool = False) -> (pd.DataFrame, Path):
    """
    Combine generation of clean CSV and resized images.
    Updates the CSV clean_data_plantvillage_segmented_all.csv so that filepath points to the generated PNGs.
    """
    clean_csv = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    output_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'segmented_clean_augmented_images'
    # List of feature columns to check for existing features
    feature_cols = FEATURE_COLS

    if clean_csv.exists():
        df_clean = pd.read_csv(clean_csv)
        # Vérifier si les features sont déjà présentes
        if not force_refresh and all(col in df_clean.columns for col in feature_cols):
            print(f"Chargement du CSV existant avec features : {clean_csv}")
            # Mettre à jour filepath pour pointer vers PNG
            df_clean['filepath'] = df_clean.apply(
                lambda r: str(output_dir / r['label'] / (Path(r['filepath']).stem + '.png')), axis=1)
            # Mettre à jour filename et extension en png
            df_clean['filename'] = df_clean['filepath'].apply(lambda p: Path(p).name)
            df_clean['extension'] = df_clean['filepath'].apply(lambda p: Path(p).suffix.lstrip('.').lower())
            # Forcer width et height pour les images redimensionnées
            df_clean['width'] = 256
            df_clean['height'] = 256
            # Spliter label en species et disease
            df_clean[['species', 'disease']] = df_clean['label'].str.split('___', expand=True)
            # Drop label column now that species and disease are extracted
            df_clean.drop(columns=['label'], inplace=True)
            # Enregistrer les modifications
            df_clean.to_csv(clean_csv, index=False)
            return df_clean, output_dir
        else:
            print("Recalcul des données et des features (rafraîchissement forcé ou colonnes manquantes)...")
    else:
        print("(Aucun CSV existant) Génération du CSV clean et des images 256x256 PNG...")
        
    print("\nGénération du CSV clean et des images 256x256 PNG en une seule passe...")
    df_clean = generate_clean_data_plantvillage_segmented_all()
    output_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'segmented_clean_images'
    # Ensure output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in df_clean['label'].unique():
        class_dir = output_dir / label
        class_dir.mkdir(exist_ok=True)
    # Resize and save images
    for idx, row in df_clean.iterrows():
        src = Path(row['filepath'])
        dst = output_dir / row['label'] / (src.stem + '.png')
        if not dst.exists():
            with Image.open(src) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
                img_resized.save(dst, 'PNG', optimize=True)
    # Update filepaths in DataFrame
    df_clean['filepath'] = df_clean.apply(
        lambda r: str(output_dir / r['label'] / (Path(r['filepath']).stem + '.png')), axis=1)
    # Update filename and extension to reflect new PNG files
    df_clean['filename'] = df_clean['filepath'].apply(lambda p: Path(p).name)
    df_clean['extension'] = df_clean['filepath'].apply(lambda p: Path(p).suffix.lstrip('.').lower())
    # Update width and height for resized images
    df_clean['width'] = 256
    df_clean['height'] = 256
    # Extract species and disease from label column
    df_clean[['species', 'disease']] = df_clean['label'].str.split('___', expand=True)
    # Drop label column now that species and disease are extracted
    df_clean.drop(columns=['label'], inplace=True)
    # Save updated CSV
    clean_csv = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    df_clean.to_csv(clean_csv, index=False)
    return df_clean, output_dir



def generate_raw_data_frame(raw_csv: Path) -> pd.DataFrame:
    """
    Generate and save raw DataFrame CSV.
    """
    df_raw = load_plantvillage_all()
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_csv, index=False)
    print(f"Raw data CSV saved: {raw_csv}")
    return df_raw


def generate_clean_images(df_raw: pd.DataFrame, clean_dir: Path, size: Tuple[int, int] = (256, 256)) -> pd.DataFrame:
    """
    Resize and save clean images. Update filepath in DataFrame.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in df_raw.iterrows():
        src = Path(row['filepath'])
        with Image.open(src) as img:
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            dest = clean_dir / f"{src.stem}.jpg"
            img_resized.save(dest, format="JPEG", quality=90)
        new_row = row.to_dict()
        new_row['filepath'] = str(dest)
        rows.append(new_row)
    df_clean = pd.DataFrame(rows)
    return df_clean


def generate_clean_dataframe(clean_csv: Path, force_refresh: bool = False) -> pd.DataFrame:
    """
    Generate or load clean DataFrame with features.
    """
    if clean_csv.exists() and not force_refresh:
        df_clean = pd.read_csv(clean_csv)
        # Ensure species/disease columns exist
        if 'species' not in df_clean.columns:
            if 'label' in df_clean.columns:
                df_clean[['species', 'disease']] = df_clean['label'].str.split('___', expand=True)
                df_clean.drop(columns=['label'], inplace=True)
            else:
                # Derive from filepath if no label column
                df_clean['species'] = df_clean['filepath'].apply(lambda p: Path(p).parent.name.split('___')[0])
                df_clean['disease'] = df_clean['filepath'].apply(lambda p: Path(p).parent.name.split('___')[1] if '___' in Path(p).parent.name else None)
        return df_clean
    # Create raw CSV and DataFrame
    raw_csv = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'raw_data_plantvillage_segmented_all.csv'
    df_raw = generate_raw_data_frame(PROJECT_ROOT, raw_csv)
    # Resize clean images
    clean_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'clean_images'
    df_clean = generate_clean_images(df_raw, clean_dir)
    # Extract features
    feats_df = extract_features_for_df(df_clean, image_col='filepath')
    df_clean = pd.concat([df_clean.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    # Split label into species and disease
    if 'label' in df_clean.columns:
        df_clean[['species', 'disease']] = df_clean['label'].str.split('___', expand=True)
        df_clean.drop(columns=['label'], inplace=True)
    # Save clean DataFrame CSV
    clean_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_csv, index=False)
    print(f"Clean data CSV saved: {clean_csv}")
    return df_clean


def generate_augmented_dataframe(aug_csv: Path, force_refresh: bool = False) -> pd.DataFrame:
    """
    Generate or load augmented DataFrame with features.
    """

    if aug_csv.exists() and not force_refresh:
        print(f"Loading existing augmented CSV: {aug_csv}")
        return pd.read_csv(aug_csv)

    # Ensure clean data is ready
    clean_csv = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    df_clean = generate_clean_dataframe(clean_csv)
    # Generate augmented images
    df_aug_imgs = augment_minority_classes_cv2(df_clean, image_col='filepath', label_col='species', force_refresh=force_refresh)
    # Extract features
    feats_df = extract_features_for_df(df_aug_imgs, image_col='filepath')
    df_aug = pd.concat([df_aug_imgs.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    # Reorder columns to match clean CSV format
    cols_order = [
        'filepath','filename','extension','file_size','width','height','mode','num_channels','aspect_ratio','is_image_valid','is_black','is_na','is_duplicate_after_first'
    ] + FEATURE_COLS + ['species','disease']
    df_aug = df_aug[cols_order]
    # Save augmented DataFrame CSV
    aug_csv.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(aug_csv, index=False)
    print(f"Augmented data CSV saved: {aug_csv}")
    return df_aug

if __name__ == "__main__":
    # Test simple des fonctions de chargement
    print("Test de load_plantvillage_all()...")
    df_all = load_plantvillage_all()
    print(f"Images totales chargées : {len(df_all)}")
    print(f"Nombre de classes : {df_all['label'].nunique()}")

    print("\nTest de load_plantvillage_five_images()...")
    df5 = load_plantvillage_five_images()
    print(f"Images totales chargées : {len(df5)}")
    print(f"Nombre de classes : {df5['label'].nunique()}")
    print("Aperçu (head) du DataFrame 5 images :")
    print(df5.head())
