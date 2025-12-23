import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

from src.helpers.helpers import PROJECT_ROOT, compute_hu_features, compute_fourier_energy, compute_hog_features, compute_pixel_ratio_and_segments

# Listes des colonnes de caractéristiques pour PlantVillage segmented
from src.data_loader.data_loader import HANDCRAFTED_FEATURE_COLS, SHAPE_DESCRIPTOR_FEATURE_COLS, FEATURE_COLS


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
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0.0
    try:
        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(cnt)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0
    except:
        eccentricity = 0.0
        major_axis = 0.0
        minor_axis = 0.0
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


def extract_and_save_features(input_csv: Path = None, output_csv: Path = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Charge le CSV combiné, extrait les caractéristiques avancées et sauvegarde dans un nouveau CSV.
    
    Args:
        input_csv: Chemin vers le fichier CSV d'entrée (combined_data_plantvillage.csv)
        output_csv: Chemin vers le fichier CSV de sortie (feature_combined_data_plantvillage.csv)
        force_refresh: Si True, recalcule les caractéristiques même si le fichier existe
        
    Returns:
        DataFrame avec toutes les données et caractéristiques
    """
    # Utiliser les chemins par défaut si non spécifiés
    if input_csv is None:
        input_csv = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'combined_data_plantvillage.csv'
    if output_csv is None:
        output_csv = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'feature_combined_data_plantvillage.csv'
    
    # Vérifier si le fichier existe déjà
    if output_csv.exists() and not force_refresh:
        print(f"Chargement du CSV existant avec caractéristiques : {output_csv}")
        return pd.read_csv(output_csv)
        
    # Charger le CSV combiné
    print(f"Chargement du CSV combiné : {input_csv}")
    df_combined = pd.read_csv(input_csv)
    
    # Extraire les caractéristiques
    print(f"Extraction des caractéristiques avancées pour {len(df_combined)} images...")
    features_df = extract_features_for_df(df_combined, image_col='filepath')
    
    # Fusionner avec le DataFrame original
    df_with_features = pd.concat([df_combined.reset_index(drop=True), 
                                 features_df.reset_index(drop=True)], axis=1)
    
    # Sauvegarder le résultat
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_with_features.to_csv(output_csv, index=False)
    print(f"CSV avec caractéristiques sauvegardé : {output_csv}")
    print(f"Nombre de caractéristiques extraites : {len(FEATURE_COLS)}")
    
    return df_with_features
