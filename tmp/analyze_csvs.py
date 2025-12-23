import pandas as pd
import os

# Define the base path
base_path = "/Users/mackjb/repository/datasciencetest_reco_plante-1/dataset/plantvillage/csv"

# Files to analyze
files = [
    "clean_data_plantvillage_segmented_all.csv",  # Original data
    "clean_with_features_data_plantvillage_segmented_all.csv",  # Original + features
    "oversampling_with_features_data_plantvillage_segmented_all.csv"  # Oversampled + features
]

# Count rows in each file
results = {}
for file in files:
    file_path = os.path.join(base_path, file)
    print(f"Counting rows in {file}...")
    try:
        # Count rows (use a chunk approach for large files)
        row_count = 0
        for chunk in pd.read_csv(file_path, chunksize=10000):
            row_count += len(chunk)
        results[file] = row_count
        print(f"  → {row_count:,} rows")
    except Exception as e:
        print(f"  → Error: {e}")

# Compare results
if "clean_data_plantvillage_segmented_all.csv" in results and "clean_with_features_data_plantvillage_segmented_all.csv" in results:
    orig_count = results["clean_data_plantvillage_segmented_all.csv"]
    clean_feat_count = results["clean_with_features_data_plantvillage_segmented_all.csv"]
    
    print("\nVerification:")
    if clean_feat_count == orig_count:
        print(f"✓ clean_with_features contains ALL original data ({clean_feat_count:,} rows)")
    else:
        print(f"! clean_with_features contains {clean_feat_count:,} rows vs. {orig_count:,} rows in original")

    if "oversampling_with_features_data_plantvillage_segmented_all.csv" in results:
        oversample_count = results["oversampling_with_features_data_plantvillage_segmented_all.csv"]
        if oversample_count >= orig_count:
            print(f"✓ oversampling_with_features contains {oversample_count:,} rows (expected to be >= original {orig_count:,})")
        else:
            print(f"! oversampling_with_features contains FEWER rows than original: {oversample_count:,} vs {orig_count:,}")