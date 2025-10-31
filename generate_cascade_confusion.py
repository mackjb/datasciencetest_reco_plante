#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère la matrice de confusion (heatmap) pour la cascade sur les classes finales.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importer depuis le script principal
sys.path.insert(0, os.path.dirname(__file__))
from train_cascade_improved import (
    list_image_files, build_label_mappings, parse_labels_from_path,
    stratified_split, make_dataset_species, make_dataset_disease, SEED
)

import tensorflow as tf
from tensorflow import keras

def main():
    # Chemins
    data_root = '/home/azureuser/localfiles/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented'
    output_dir = 'outputs/cascade_improved'
    
    print("[INFO] Chargement des modèles...")
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
    
    # Arguments
    img_size = (256, 256)
    batch_size = 32
    num_species = len(species_to_idx)
    num_diseases = len(disease_to_idx)
    
    # STEP 1: Prédire espèces
    print("\n[INFO] Prédiction des espèces...")
    test_ds_species = make_dataset_species(test_paths, test_species_labels,
                                          img_size, batch_size, num_species, False, SEED)
    species_probs = species_model.predict(test_ds_species, verbose=1)
    species_pred = np.argmax(species_probs, axis=1)
    
    # STEP 2: Prédire maladies avec espèces PRÉDITES
    print("\n[INFO] Prédiction des maladies avec espèces PRÉDITES...")
    test_ds_disease = make_dataset_disease(test_paths, species_pred, test_disease_labels,
                                          img_size, batch_size, num_species, num_diseases, False, SEED)
    disease_probs = disease_model.predict(test_ds_disease, verbose=1)
    disease_pred = np.argmax(disease_probs, axis=1)
    
    # Créer classes finales
    true_final_classes = [f"{idx_to_species[sp]}___{idx_to_disease[dis]}" 
                         for sp, dis in zip(test_species_labels, test_disease_labels)]
    pred_final_classes = [f"{idx_to_species[sp]}___{idx_to_disease[dis]}" 
                         for sp, dis in zip(species_pred, disease_pred)]
    
    # Obtenir toutes les classes uniques (triées)
    all_classes = sorted(set(true_final_classes))
    print(f"\n[INFO] Nombre de classes finales: {len(all_classes)}")
    
    # Matrice de confusion
    print("\n[INFO] Génération de la matrice de confusion...")
    cm = confusion_matrix(true_final_classes, pred_final_classes, labels=all_classes)
    
    # Créer un DataFrame pour une meilleure visualisation
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    
    # Sauvegarder la matrice brute
    cm_df.to_csv(os.path.join(output_dir, 'cascade_confusion_matrix.csv'))
    print(f"[INFO] Matrice sauvegardée: cascade_confusion_matrix.csv")
    
    # Créer la heatmap - Version complète (grande)
    print("\n[INFO] Génération de la heatmap complète...")
    plt.figure(figsize=(28, 24))
    
    # Utiliser un masque pour les zéros (plus lisible)
    mask = cm == 0
    
    sns.heatmap(cm_df, annot=False, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Nombre de prédictions'},
                square=True, linewidths=0.5, linecolor='gray',
                mask=mask, vmin=0, vmax=cm.max())
    
    plt.title('Matrice de Confusion CASCADE (Espèce × Maladie)\n38 Classes Finales', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Classe Prédite', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Vraie', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'cascade_confusion_heatmap_full.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Heatmap complète sauvegardée: cascade_confusion_heatmap_full.png")
    plt.close()
    
    # Créer une version normalisée (pourcentages)
    print("\n[INFO] Génération de la heatmap normalisée...")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=all_classes, columns=all_classes)
    
    plt.figure(figsize=(28, 24))
    mask_norm = cm_normalized < 0.001  # Masquer valeurs très faibles
    
    sns.heatmap(cm_norm_df, annot=False, fmt='.2%', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Proportion (%)'},
                square=True, linewidths=0.5, linecolor='gray',
                mask=mask_norm, vmin=0, vmax=1)
    
    plt.title('Matrice de Confusion CASCADE - Normalisée\n(% par classe vraie)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Classe Prédite', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Vraie', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    output_path_norm = os.path.join(output_dir, 'cascade_confusion_heatmap_normalized.png')
    plt.savefig(output_path_norm, dpi=150, bbox_inches='tight')
    print(f"[INFO] Heatmap normalisée sauvegardée: cascade_confusion_heatmap_normalized.png")
    plt.close()
    
    # Créer une version avec uniquement les erreurs (hors diagonale)
    print("\n[INFO] Génération de la heatmap des erreurs...")
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)  # Mettre la diagonale à 0
    
    if cm_errors.sum() > 0:
        cm_errors_df = pd.DataFrame(cm_errors, index=all_classes, columns=all_classes)
        
        plt.figure(figsize=(28, 24))
        mask_errors = cm_errors == 0
        
        sns.heatmap(cm_errors_df, annot=False, fmt='d', cmap='Reds', 
                    cbar_kws={'label': 'Nombre d\'erreurs'},
                    square=True, linewidths=0.5, linecolor='gray',
                    mask=mask_errors, vmin=0)
        
        plt.title('Matrice des Erreurs CASCADE\n(Diagonale exclue)', 
                  fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Classe Prédite (erreur)', fontsize=14, fontweight='bold')
        plt.ylabel('Classe Vraie', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        output_path_errors = os.path.join(output_dir, 'cascade_confusion_errors_only.png')
        plt.savefig(output_path_errors, dpi=150, bbox_inches='tight')
        print(f"[INFO] Heatmap des erreurs sauvegardée: cascade_confusion_errors_only.png")
        plt.close()
        
        # Statistiques sur les erreurs
        print(f"\n[INFO] Nombre total d'erreurs: {cm_errors.sum()}")
        print(f"[INFO] Accuracy cascade: {(cm.trace() / cm.sum()) * 100:.2f}%")
    
    # Identifier les paires de classes les plus confondues
    print("\n" + "="*80)
    print("TOP 10 - CONFUSIONS LES PLUS FRÉQUENTES")
    print("="*80)
    
    errors_list = []
    for i, true_class in enumerate(all_classes):
        for j, pred_class in enumerate(all_classes):
            if i != j and cm[i, j] > 0:
                errors_list.append({
                    'True': true_class,
                    'Predicted': pred_class,
                    'Count': cm[i, j],
                    'Percent_of_true': (cm[i, j] / cm[i].sum()) * 100
                })
    
    if errors_list:
        errors_df = pd.DataFrame(errors_list).sort_values('Count', ascending=False)
        print(errors_df.head(10).to_string(index=False))
        errors_df.to_csv(os.path.join(output_dir, 'cascade_top_confusions.csv'), index=False)
        print(f"\n[INFO] Top confusions sauvegardées: cascade_top_confusions.csv")
    else:
        print("Aucune confusion! Modèle parfait 🎉")
    
    print("\n" + "="*80)
    print("GÉNÉRATION TERMINÉE")
    print("="*80)
    print(f"Fichiers générés dans: {output_dir}/")
    print("  - cascade_confusion_matrix.csv (données brutes)")
    print("  - cascade_confusion_heatmap_full.png (heatmap complète)")
    print("  - cascade_confusion_heatmap_normalized.png (% par classe)")
    print("  - cascade_confusion_errors_only.png (erreurs uniquement)")
    print("  - cascade_top_confusions.csv (top confusions)")

if __name__ == '__main__':
    main()
