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
default_dataset: str = 'plantvillage_5images/segmented'



def dataset_to_dataframe(subfolder: str = default_dataset) -> pd.DataFrame:
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
    
    # Sauvegarder automatiquement dans le fichier CSV
    csv_path = PROJECT_ROOT / 'dataset' / 'plantvillage' / 'csv' / 'raw_data_plantvillage_segmented_all.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # Créer le répertoire si nécessaire
    df.to_csv(csv_path, index=False)
    print(f"Données sauvegardées dans {csv_path}")
    
    return df
