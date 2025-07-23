from src.data_loader import (
    dataset_to_clean_dataframe,
    generate_plantvillage_images,
    extract_and_save_features,
)
from src.data_loader.data_loader import PROJECT_ROOT


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

# 3. Extraire les caractéristiques avancées des images
print("\n=== EXTRACTION DES CARACTÉRISTIQUES AVANCÉES ===")
df_features = extract_and_save_features(force_refresh=True)
print(f"\nDataFrame avec caractéristiques : {df_features.shape[0]} lignes x {df_features.shape[1]} colonnes")

# Afficher un aperçu des caractéristiques extraites
feature_cols = [col for col in df_features.columns if col not in df_combined.columns or col == 'filepath']
print(f"Nombre de caractéristiques extraites : {len(feature_cols)-1}")  # -1 pour exclure filepath

# Afficher quelques statistiques sur les caractéristiques
print("\nStatistiques des caractéristiques de forme :")
shape_features = [col for col in feature_cols if col in ['area', 'perimeter', 'circularity', 'solidity', 'fractal_dimension'] or col == 'filepath']
if len(shape_features) > 1:  # S'assurer qu'il y a des caractéristiques autres que filepath
    print(df_features[shape_features].describe().round(2))

# Distribution des espèces et maladies
print("\nDistribution des espèces :")
print(df_features['species'].value_counts().head())
print("\nDistribution des maladies :")
print(df_features['disease'].value_counts().head())
