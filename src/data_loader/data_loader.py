import glob
import pandas as pd
from pathlib import Path

from src.helpers.helpers import PROJECT_ROOT

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
                records.append({
                    'filepath': str(img_path),
                    'label': class_name
                })
    # Création du DataFrame
    df = pd.DataFrame(records)
    return df


def load_plantvillage_all() -> pd.DataFrame:
    """
    Charge l'intégralité du dataset PlantVillage.

    :return: pandas.DataFrame avec colonnes 'filepath' et 'label'.
    """
    return _load_dataset('plantvillage dataset')


def load_plantvillage_five_images() -> pd.DataFrame:
    """
    Charge un sous-ensemble du dataset PlantVillage contenant cinq images par classe.

    :return: pandas.DataFrame avec colonnes 'filepath' et 'label'.
    """
    return _load_dataset('plantvillage_5images')


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
