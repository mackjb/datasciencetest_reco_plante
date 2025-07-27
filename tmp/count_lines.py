#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

# Define CSV files and their paths
csv_dir = "/Users/mackjb/repository/datasciencetest_reco_plante-1/dataset/plantvillage/csv"
csv_files = [
    "clean_data_plantvillage_segmented_all.csv",
    "clean_with_features_data_plantvillage_segmented_all.csv",
    "oversampling_with_features_data_plantvillage_segmented_all.csv"
]

print("Counting rows in CSV files...")
print("=" * 50)

# For each CSV file, count rows and print result
for csv_file in csv_files:
    full_path = os.path.join(csv_dir, csv_file)
    try:
        # Simple line counting (subtract 1 for header)
        with open(full_path, 'r') as f:
            line_count = sum(1 for _ in f) - 1
        
        # Print result
        print(f"{csv_file}: {line_count:,} rows")
        
    except Exception as e:
        print(f"Error counting {csv_file}: {str(e)}")

print("=" * 50)
print("Analysis complete!")