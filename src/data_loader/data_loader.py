import glob
import pandas as pd
import numpy as np
from pathlib import Path

from src.helpers.helpers import PROJECT_ROOT, compute_hu_features, compute_fourier_energy, compute_hog_features, compute_pixel_ratio_and_segments
from PIL import Image, ImageStat
import hashlib

def is_black_image(image_path, threshold=10):
    """
    Returns True if the image is mostly black based on the mean grayscale value.
    """
    with Image.open(image_path).convert("L") as img_gray:
        stat = ImageStat.Stat(img_gray)
        mean_val = stat.mean[0]
    return mean_val < threshold

def is_image_valid(image_path: str) -> bool:
    """Vérifie que l’image n’est pas corrompue."""
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
    df['is_duplicate'] = df['hash'].duplicated(keep=False)
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


def generate_clean_data_plantvillage_segmented_all() -> pd.DataFrame:
    """
    Charge, nettoie, enrichit et sauvegarde le dataset PlantVillage segmented.
    Retourne le DataFrame clean.
    """
    df_raw = load_plantvillage_all()
    # Filtrage des NA et duplicates
    df_clean = df_raw[~(df_raw['is_na'] | df_raw['is_duplicate'])].drop_duplicates(subset='filepath').copy()

    def extract_features(row):
        path = row['filepath']
        try:
            with Image.open(path) as img:
                gray = np.array(img.convert('L'))
            hu = compute_hu_features(gray)
            fourier = compute_fourier_energy(gray)
            hog = compute_hog_features(gray)
            pix = compute_pixel_ratio_and_segments(gray)
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
            })
        except Exception:
            return pd.Series({col: None for col in [
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
                'leaf_segments'
            ]})

    feats_df = df_clean.apply(extract_features, axis=1)
    df_clean = pd.concat([df_clean, feats_df], axis=1)

    clean_csv = PROJECT_ROOT / 'clean_data_plantvillage_segmented_all.csv'
    df_clean.to_csv(clean_csv, index=False)
    return df_clean


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
