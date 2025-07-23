from src.data_loader.data_loader import (
    dataset_to_clean_dataframe,
    generate_plantvillage_images,
    PROJECT_ROOT,
)


# clean_csv = PROJECT_ROOT/"dataset"/"plantvillage"/"csv"/"clean_data_plantvillage_segmented_all.csv"
# augmented_csv   = PROJECT_ROOT/"dataset"/"plantvillage"/"csv"/"augmented_minority_cv2.csv"

# 1. Charger et nettoyer les données
df = dataset_to_clean_dataframe()
print(df.head())
print(df.shape)

# 2. Générer les images nettoyées ET augmentées (pipeline complet)
df_clean, df_combined = generate_plantvillage_images(force_refresh=True)
print(f"\nImages propres générées : {len(df_clean)}")
print(f"Images après augmentation : {len(df_combined)}")
print(f"Colonnes disponibles : {df_clean.columns.tolist()}")

# Afficher la distribution des classes après augmentation
print("\nDistribution des classes après augmentation :")
print(df_combined['label'].value_counts().sort_values(ascending=False).head(10))
