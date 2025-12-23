#!/bin/bash
set -e


echo "Creating lightweight deployment branch..."

# Delete deploy branch if exists
git branch -D deploy 2>/dev/null || true

# Switch to orphan branch
git checkout --orphan deploy

# Unstage all files (keep them on disk)
git reset

# Configure LFS properly
git lfs install
git lfs track "*.png" "*.jpg" "*.jpeg" "*.csv"
git add .gitattributes



# Add essential files
echo "Adding essential files..."
git add Streamlit/ src/ Dockerfile requirements.txt README.md figures/

# Ensure minimal CSV exists
if [ ! -f dataset/plantvillage/csv/minimal_eda_data.csv ]; then
    echo "Regenerating minimal CSV..."
    python3 -c "import pandas as pd; import os; 
if os.path.exists('dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv'):
    df = pd.read_csv('dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv'); 
    df[['nom_plante', 'Est_Saine', 'nom_maladie']].to_csv('dataset/plantvillage/csv/minimal_eda_data.csv', index=False); 
    print('CSV generated.')
else:
    print('Source CSV not found!')"
fi

# Add specific dataset CSV (force add in case it's ignored)
git add -f dataset/plantvillage/csv/minimal_eda_data.csv

# Add specific results needed by Streamlit
git add results/feature_ranking.csv
git add results/Machine_Learning/svm_rbf_baseline_features_selected/plots/baseline/confusion_matrix.png
git add results/Deep_Learning/archi1_outputs_mono_disease_effv2s_256_color_split/class_counts.csv
# Add whitelisted Deep Learning assets
if [ -f deploy_files_whitelist.txt ]; then
    echo "Adding whitelisted assets..."
    # Handle spaces in filenames if any (though looking at the list they seem safe, but xargs -d '\n' is safer)
    cat deploy_files_whitelist.txt | tr '\n' '\0' | xargs -0 git add -f
else
    echo "Warning: whitelist not found!"
fi

# Commit
git commit -m "Deploy lightweight app version"

echo "------------------------------------------------"
echo "Deployment branch 'deploy' is ready!"
echo "Size of this commit:"
git count-objects -vH
echo "------------------------------------------------"
echo "You can now push with:"
echo "git push -f space deploy:main"
