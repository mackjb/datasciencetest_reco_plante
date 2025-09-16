# Rapport de Comparaison des Modèles AutoML

## Informations Générales
- Date du rapport: 2025-09-16 15:51:57
- Tâche Espèces: 20250916_153446
- Tâche Maladies: 20250916_154829

## Meilleurs Modèles

### Classification d'Espèces
- **Modèle**: Light Gradient Boosting Machine
- **Score F1**: 0.8997
- **Fichier du modèle**: `results/automl_especes/best_model_20250916_153446.pkl`

### Classification de Maladies
- **Modèle**: Light Gradient Boosting Machine
- **Score F1**: 0.8148
- **Fichier du modèle**: `results/automl_maladies/best_model_20250916_154829.pkl`

## Comparaison des Performances

Les graphiques suivants comparent les performances des modèles pour les deux tâches:

### ACCURACY - Comparaison
![accuracy](figures/comparison/accuracy_comparison.png)

### F1 - Comparaison
![f1](figures/comparison/f1_comparison.png)

### AUC - Comparaison
![auc](figures/comparison/auc_comparison.png)

### RECALL - Comparaison
![recall](figures/comparison/recall_comparison.png)

### PRECISION - Comparaison
![precision](figures/comparison/precision_comparison.png)

## Recommandations

1. **Modèle pour la classification d'espèces**: Light Gradient Boosting Machine (F1: 0.8997)
2. **Modèle pour la classification de maladies**: Light Gradient Boosting Machine (F1: 0.8148)
3. **Observations**: Les modèles LightGBM et Random Forest offrent les meilleures performances pour les deux tâches. LightGBM est légèrement plus performant mais peut être plus gourmand en ressources.

