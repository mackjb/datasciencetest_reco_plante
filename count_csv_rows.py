#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to count rows in CSV files and verify data completeness
"""

import pandas as pd
import os

def count_csv_rows(csv_path):
    """Count number of rows in a CSV file"""
    print(f"Counting rows in: {os.path.basename(csv_path)}")
    try:
        # Use a chunk-based approach for large files
        chunk_size = 10000
        row_count = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            row_count += len(chunk)
        return row_count
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return -1

def main():
    base_path = "dataset/plantvillage/csv"
    
    # Files to analyze
    files = [
        "clean_data_plantvillage_segmented_all.csv",  # Original data
        "clean_with_features_data_plantvillage_segmented_all.csv",  # Original + features
        "oversampling_with_features_data_plantvillage_segmented_all.csv"  # Oversampled + features
    ]
    
    print("=== CSV Row Count Analysis ===")
    results = {}
    
    for file in files:
        file_path = os.path.join(base_path, file)
        row_count = count_csv_rows(file_path)
        results[file] = row_count
        
    print("\n=== Results ===")
    for file, count in results.items():
        print(f"{file}: {count:,} rows")
    
    # Check if feature files contain all original data
    if results.get("clean_data_plantvillage_segmented_all.csv", 0) > 0:
        original_count = results["clean_data_plantvillage_segmented_all.csv"]
        clean_with_features_count = results.get("clean_with_features_data_plantvillage_segmented_all.csv", 0)
        
        print("\n=== Verification ===")
        if clean_with_features_count == original_count:
            print(f"✓ clean_with_features file contains ALL original data ({clean_with_features_count:,} rows)")
        else:
            print(f"✗ clean_with_features file contains {clean_with_features_count:,} rows " +
                 f"vs. {original_count:,} rows in original data")
        
        # The oversampled file should have more rows than original
        oversampling_count = results.get("oversampling_with_features_data_plantvillage_segmented_all.csv", 0)
        if oversampling_count >= original_count:
            print(f"✓ oversampling_with_features file contains {oversampling_count:,} rows " +
                 f"(expected to be >= original {original_count:,} rows)")
        else:
            print(f"✗ oversampling_with_features file contains FEWER rows than original: " +
                 f"{oversampling_count:,} vs {original_count:,}")

if __name__ == "__main__":
    main()