#!/bin/bash

# Base directory
CSV_DIR="/Users/mackjb/repository/datasciencetest_reco_plante-1/dataset/plantvillage/csv"

# Files to analyze
FILES=(
  "clean_data_plantvillage_segmented_all.csv"
  "clean_with_features_data_plantvillage_segmented_all.csv"
  "oversampling_with_features_data_plantvillage_segmented_all.csv"
)

echo "=== CSV Row Count Analysis ==="
echo

# Count rows in each file (subtract 1 for header)
declare -A counts
for file in "${FILES[@]}"; do
  if [ -f "$CSV_DIR/$file" ]; then
    # Count lines and subtract 1 for header
    line_count=$(wc -l < "$CSV_DIR/$file")
    row_count=$((line_count - 1))
    counts[$file]=$row_count
    echo "$file: $row_count rows"
  else
    echo "$file: File not found"
  fi
done

echo
echo "=== Verification ==="

# Compare row counts
original_count=${counts["clean_data_plantvillage_segmented_all.csv"]}
features_count=${counts["clean_with_features_data_plantvillage_segmented_all.csv"]}
oversampled_count=${counts["oversampling_with_features_data_plantvillage_segmented_all.csv"]}

# Check if clean with features contains all original data
if [ "$features_count" -eq "$original_count" ]; then
  echo "✓ clean_with_features file contains ALL original data ($features_count rows)"
else
  echo "! clean_with_features file contains $features_count rows vs. $original_count rows in original data"
  percent=$(echo "scale=2; 100*$features_count/$original_count" | bc)
  echo "  → That's $percent% of the original data"
fi

# Check if oversampled data has more rows than original
if [ "$oversampled_count" -ge "$original_count" ]; then
  echo "✓ oversampling_with_features file contains $oversampled_count rows (expected to be >= original $original_count rows)"
  increase=$(echo "scale=2; 100*($oversampled_count-$original_count)/$original_count" | bc)
  echo "  → That's a $increase% increase from the original data"
else
  echo "! oversampling_with_features file contains FEWER rows than original: $oversampled_count vs $original_count"
  percent=$(echo "scale=2; 100*$oversampled_count/$original_count" | bc)
  echo "  → That's only $percent% of the original data"
fi