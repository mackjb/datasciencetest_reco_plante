from IPython.display import display
from src.helpers.helpers import PROJECT_ROOT
from src.data_loader.data_loader import load_plantvillage_all, load_plantvillage_five_images, generate_raw_data_plantvillage_segmented_all, generate_clean_data_plantvillage_segmented_all, generate_segmented_clean_augmented_images
import time
import hashlib
import pandas as pd







if __name__ == "__main__":
    # print("\nTest de load_plantvillage_all()...")
    # t0 = time.time()
    # df_all = load_plantvillage_all()
    # dt0 = time.time() - t0
    # print(f"load_all: {len(df_all)} images en {dt0:.2f}s, classes: {df_all['label'].nunique()}")

    # print("\nTest de load_plantvillage_five_images()...")
    # t1 = time.time()
    # df5 = load_plantvillage_five_images()
    # dt1 = time.time() - t1
    # print(f"load_5_images: {len(df5)} images en {dt1:.2f}s, classes: {df5['label'].nunique()}")
    # print("Aperçu (head) du DataFrame 5 images :")
    # print(df5.head())

    # Génération du CSV complet raw_data_plantvillage_segmented_all.csv
    print("\nGénération du CSV complet raw_data_plantvillage_segmented_all.csv...")
    df_raw = generate_raw_data_plantvillage_segmented_all()
    print(df_raw.head())

    # Délégation à data_loader pour génération du CSV clean
    print("\nGénération du CSV clean via data_loader...")
    df_clean = generate_clean_data_plantvillage_segmented_all()
    print(df_clean.head())
    
    # Génération des images standardisées 256x256 PNG
    print("\nGénération des images standardisées 256x256 PNG...")
    output_dir = generate_segmented_clean_augmented_images()
    print(f"Images standardisées générées dans : {output_dir}")

    # # Séparation en clean vs outliers
    # flags = df_all['is_na'] | df_all['is_duplicate'] | df_all['is_fail_segmented']
    # df_clean = df_all[~flags]
    # df_outliers = df_all[flags]
    # clean_csv = PROJECT_ROOT / 'plantvillage_clean.csv'
    # outliers_csv = PROJECT_ROOT / 'plantvillage_outliers.csv'
    # df_clean.to_csv(clean_csv, index=False)
    # df_outliers.to_csv(outliers_csv, index=False)
    # print(f"CSV clean enregistré: {clean_csv}")
    # print(f"CSV outliers enregistré: {outliers_csv}")
