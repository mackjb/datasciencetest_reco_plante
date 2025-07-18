from IPython.display import display
from src.helpers.helpers import PROJECT_ROOT
from src.data_loader.data_loader import load_plantvillage_all, load_plantvillage_five_images
import time
import hashlib







if __name__ == "__main__":
    # print("\nTest de load_plantvillage_all()...")
    # t0 = time.time()
    # df_all = load_plantvillage_all()
    # dt0 = time.time() - t0
    # print(f"load_all: {len(df_all)} images en {dt0:.2f}s, classes: {df_all['label'].nunique()}")

    print("\nTest de load_plantvillage_five_images()...")
    t1 = time.time()
    df5 = load_plantvillage_five_images()
    dt1 = time.time() - t1
    print(f"load_5_images: {len(df5)} images en {dt1:.2f}s, classes: {df5['label'].nunique()}")
    print("Aperçu (head) du DataFrame 5 images :")
    print(df5.head())

    # # Génération du CSV complet plantvillage_segmented_all.csv
    # print("\nGénération du CSV complet plantvillage_segmented_all.csv...")
    # df_all = load_plantvillage_all()
    # # Calcul des hash pour détecter les doublons
    # df_all['hash'] = df_all['filepath'].apply(lambda p: hashlib.md5(open(p,'rb').read()).hexdigest())
    # # Flags
    # df_all['is_duplicate'] = df_all['hash'].duplicated(keep=False)
    # df_all['is_fail_segmented'] = df_all['leaf_segments'].isna() | (df_all['leaf_segments'] == 0)
    # df_all.drop(columns=['hash'], inplace=True)
    # out_csv = PROJECT_ROOT / 'plantvillage_segmented_all.csv'
    # df_all.to_csv(out_csv, index=False)
    # print(f"CSV enregistré: {out_csv}")

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
