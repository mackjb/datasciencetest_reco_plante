#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ré-évaluation de la cascade avec le nouveau calcul de F1.
Utilise les modèles déjà entraînés.
"""
import json
import os
import sys

# Importer depuis le script principal
sys.path.insert(0, os.path.dirname(__file__))
from train_cascade_improved import (
    list_image_files, build_label_mappings, parse_labels_from_path,
    stratified_split, evaluate_cascade, evaluate_oracle, SEED
)

import tensorflow as tf
from tensorflow import keras

def main():
    # Chemins
    data_root = '/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented'
    output_dir = 'outputs/cascade_improved'
    
    print("[INFO] Chargement des modèles entraînés...")
    species_model = keras.models.load_model(os.path.join(output_dir, 'species_model.keras'))
    disease_model = keras.models.load_model(os.path.join(output_dir, 'disease_model_global.keras'))
    
    print("[INFO] Chargement du dataset...")
    files = list_image_files(data_root)
    
    # Mappings
    (species_to_idx, idx_to_species,
     disease_to_idx, idx_to_disease,
     species_disease_mappings) = build_label_mappings(files)
    
    # Labels
    species_labels = []
    disease_labels = []
    for fp in files:
        species, disease = parse_labels_from_path(fp)
        species_labels.append(species_to_idx[species])
        disease_labels.append(disease_to_idx[disease])
    
    # Split
    combined_labels = [f"{s}_{d}" for s, d in zip(species_labels, disease_labels)]
    (train_paths, train_combined,
     val_paths, val_combined,
     test_paths, test_combined) = stratified_split(files, combined_labels, seed=SEED)
    
    test_species_labels = [species_labels[files.index(p)] for p in test_paths]
    test_disease_labels = [disease_labels[files.index(p)] for p in test_paths]
    
    print(f"[INFO] Test set: {len(test_paths)} images")
    
    # Arguments factices pour l'évaluation
    class Args:
        img_size = [256, 256]
        batch_size = 32
    args = Args()
    
    # ORACLE
    print("\n" + "="*60)
    print("RÉ-ÉVALUATION ORACLE")
    print("="*60)
    oracle_results = evaluate_oracle(
        disease_model,
        test_paths, test_species_labels, test_disease_labels,
        len(species_to_idx), len(disease_to_idx),
        output_dir, args
    )
    
    # CASCADE
    print("\n" + "="*60)
    print("RÉ-ÉVALUATION CASCADE (NOUVEAU CALCUL F1)")
    print("="*60)
    cascade_results = evaluate_cascade(
        species_model, disease_model,
        test_paths, test_species_labels, test_disease_labels,
        len(species_to_idx), len(disease_to_idx),
        idx_to_species, idx_to_disease,
        output_dir, args
    )
    
    print("\n" + "="*60)
    print("RÉ-ÉVALUATION TERMINÉE")
    print("="*60)
    print("Résultats mis à jour dans outputs/cascade_improved/")

if __name__ == '__main__':
    main()
