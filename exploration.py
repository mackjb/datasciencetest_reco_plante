from IPython.display import display
from src.helpers.helpers import PROJECT_ROOT
from src.data_loader.data_loader import load_plantvillage_all, load_plantvillage_five_images, generate_raw_data_plantvillage_segmented_all, generate_clean_data_plantvillage_segmented_all, generate_clean_and_resized, generate_segmented_clean_augmented_images
import time
import hashlib
import pandas as pd
from pathlib import Path
from PIL import Image








if __name__ == "__main__":

    # Génération du CSV complet raw_data_plantvillage_segmented_all.csv
    print("\nGénération du CSV complet raw_data_plantvillage_segmented_all.csv...")
    df_raw = generate_raw_data_plantvillage_segmented_all()
    print(df_raw.head())

    # Génération du CSV clean et des images 256x256 PNG en une seule méthode
    df_clean, output_dir = generate_clean_and_resized()
    print(df_clean.head())
    print(f"Clean CSV mis à jour et images PNG générées dans : {output_dir}")



