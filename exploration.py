from src.data_loader.data_loader import (
    dataset_to_clean_dataframe,
    PROJECT_ROOT,
)


# clean_csv = PROJECT_ROOT/"dataset"/"plantvillage"/"csv"/"clean_data_plantvillage_segmented_all.csv"
# augmented_csv   = PROJECT_ROOT/"dataset"/"plantvillage"/"csv"/"augmented_minority_cv2.csv"

df = dataset_to_clean_dataframe()
print(df.head())
print(df.shape)
