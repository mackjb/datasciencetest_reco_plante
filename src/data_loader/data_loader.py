import glob
import pandas as pd
from pathlib import Path

from src.helpers.helpers import PROJECT_ROOT, compute_hu_features, compute_fourier_energy, compute_hog_features
from PIL import Image

# Base path for the PlantVillage dataset directories
# Utilise la constante PROJECT_ROOT pour localiser le dossier data
data_root: Path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'data'

def _load_dataset(subfolder: str) -> pd.DataFrame:
    """
    Charge un sous-dossier du dataset PlantVillage et retourne un DataFrame
    avec deux colonnes : 'filepath' et 'label'.

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
                    width, height, mode, num_channels, aspect_ratio = None, None, None, None, None
                # Initialisation par défaut des features spectrales et HOG
                energie_basse = energie_moyenne = energie_haute = None
                hog_mean = hog_std = None
                try:
                    hu_feats = compute_hu_features(img_path)
                    try:
                        fourier_feats = compute_fourier_energy(img_path)
                        energie_basse = fourier_feats['energie_basse_forme_feuille']
                        energie_moyenne = fourier_feats['energie_moyenne_texture_veines']
                        energie_haute = fourier_feats['energie_haute_details_maladie']
                    except Exception:
                        energie_basse = energie_moyenne = energie_haute = None
                    try:
                        hog_feats = compute_hog_features(img_path)
                        hog_mean = hog_feats['hog_moyenne_contours_forme']
                        hog_std = hog_feats['hog_ecarttype_texture']
                    except Exception:
                        hog_mean = hog_std = None

                    hu_phi1 = hu_feats['phi1_distingue_large_vs_etroit']
                    hu_phi2 = hu_feats['phi2_distinction_elongation_forme']
                    hu_phi3 = hu_feats['phi3_asymetrie_maladie']
                    hu_phi4 = hu_feats['phi4_symetrie_diagonale_forme']
                    hu_phi5 = hu_feats['phi5_concavite_extremites']
                    hu_phi6 = hu_feats['phi6_decalage_torsion_maladie']
                    hu_phi7 = hu_feats['phi7_asymetrie_complexe']
                except Exception:
                    hu_phi1 = hu_phi2 = hu_phi3 = hu_phi4 = hu_phi5 = hu_phi6 = hu_phi7 = None
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
                    'phi1_distingue_large_vs_etroit': hu_phi1,
                    'phi2_distinction_elongation_forme': hu_phi2,
                    'phi3_asymetrie_maladie': hu_phi3,
                    'phi4_symetrie_diagonale_forme': hu_phi4,
                    'phi5_concavite_extremites': hu_phi5,
                    'phi6_decalage_torsion_maladie': hu_phi6,
                    'phi7_asymetrie_complexe': hu_phi7,
                    'energie_basse_forme_feuille': energie_basse,
                    'energie_moyenne_texture_veines': energie_moyenne,
                    'energie_haute_details_maladie': energie_haute,
                    'hog_moyenne_contours_forme': hog_mean,
                    'hog_ecarttype_texture': hog_std,

                })
    # Création du DataFrame
    df = pd.DataFrame(records)
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
