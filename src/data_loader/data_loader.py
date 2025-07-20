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

    return df_clean


def generate_clean_and_resized() -> (pd.DataFrame, Path):
    """
    Combine generation of clean CSV and resized images.
    Updates the CSV clean_data_plantvillage_segmented_all.csv so that filepath points to the generated PNGs.
    """
    print("\nGénération du CSV clean et des images 256x256 PNG en une seule passe...")
    df_clean = generate_clean_data_plantvillage_segmented_all()
    output_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'segmented_clean_augmented_images'
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
    # Save updated CSV
    clean_csv = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    df_clean.to_csv(clean_csv, index=False)
    return df_clean, output_dir


def generate_segmented_clean_augmented_images():
    """Deprecated: use generate_clean_and_resized()."""
    raise NotImplementedError('generate_segmented_clean_augmented_images is deprecated; use generate_clean_and_resized')
    """
    Lit le fichier clean_data_plantvillage_segmented_all.csv et génère des images
    standardisées 256x256 en format PNG dans le répertoire segmented_clean_augmented_images.
    """
    # Chemins des fichiers et répertoires
    csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'clean_data_plantvillage_segmented_all.csv'
    output_dir = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'segmented_clean_augmented_images'
    
    # Vérifier que le CSV existe
    if not csv_path.exists():
        raise FileNotFoundError(f"Le fichier CSV {csv_path} n'existe pas. Exécutez d'abord generate_clean_data_plantvillage_segmented_all().")
    
    # Charger le CSV
    print(f"Chargement du CSV depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Nombre d'images à traiter : {len(df)}")
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer les sous-répertoires par classe
    for label in df['label'].unique():
        class_dir = output_dir / label
        class_dir.mkdir(exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    print("Traitement des images...")
    for idx, row in df.iterrows():
        try:
            # Chemin de l'image source
            source_path = Path(row['filepath'])
            
            # Nom du fichier de sortie (changement d'extension vers PNG)
            output_filename = Path(row['filename']).stem + '.png'
            output_path = output_dir / row['label'] / output_filename
            
            # Éviter de retraiter si l'image existe déjà
            if output_path.exists():
                processed_count += 1
                continue
            
            # Charger et redimensionner l'image
            with Image.open(source_path) as img:
                # Convertir en RGB si nécessaire (pour assurer la compatibilité PNG)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionner à 256x256 avec un bon algorithme de rééchantillonnage
                img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
                
                # Sauvegarder en PNG (format lossless)
                img_resized.save(output_path, 'PNG', optimize=True)
            
            processed_count += 1
            
            # Affichage du progrès
            if processed_count % 100 == 0:
                print(f"Traité {processed_count}/{len(df)} images...")
                
        except Exception as e:
            error_count += 1
            print(f"Erreur lors du traitement de {row['filepath']}: {e}")
            continue
    
    print(f"\nTraitement terminé !")
    print(f"Images traitées avec succès : {processed_count}")
    print(f"Erreurs : {error_count}")
    print(f"Répertoire de sortie : {output_dir}")
    
    return output_dir


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
