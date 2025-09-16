# Rapport de Sélection des Modèles

## 1. Résumé Exécutif

Ce document présente les résultats de l'exploration des modèles de classification pour la reconnaissance de plantes et de maladies. L'objectif était d'identifier les modèles les plus performants pour deux tâches distinctes :
1. Classification des espèces de plantes
2. Classification des maladies des plantes

## 2. Méthodologie

### 2.1 Données Utilisées
- **Source** : PlantVillage Dataset (version nettoyée et segmentée)
- **Taille de l'ensemble de données** : 54,251 échantillons
  - Échantillons de maladies : 39,185
  - Échantillons sains : 15,066

### 2.2 Approche
- Utilisation de PyCaret pour l'exploration automatisée de modèles
- Test de 5 algorithmes de classification par tâche
- Métrique d'évaluation principale : Score F1
- Validation croisée avec 3 folds

## 3. Résultats par Tâche

### 3.1 Classification d'Espèces

#### Meilleur Modèle : Light Gradient Boosting Machine
- **Score F1** : 0.8997
- **Autres Métriques** :
  - Précision : 0.9003
  - Rappel : 0.9003
  - AUC : 0.9929
  - Temps d'entraînement : 5 minutes 30 secondes

#### Classement Complet des Modèles :
1. LightGBM (F1: 0.8997)
2. Random Forest (F1: 0.8718)
3. Extra Trees (F1: 0.8687)
4. K-Nearest Neighbors (F1: 0.7680)
5. LDA (F1: 0.7421)

### 3.2 Classification de Maladies

#### Meilleur Modèle : Light Gradient Boosting Machine
- **Score F1** : 0.8148
- **Autres Métriques** :
  - Précision : 0.8160
  - Rappel : 0.8142
  - AUC : 0.9867
  - Temps d'entraînement : 4 minutes 57 secondes

#### Classement Complet des Modèles :
1. LightGBM (F1: 0.8148)
2. Random Forest (F1: 0.7773)
3. Extra Trees (F1: 0.7743)
4. K-Nearest Neighbors (F1: 0.6489)
5. LDA (F1: 0.6252)

## 4. Analyse Comparative

### 4.1 Performances Globales
- Les modèles basés sur le boosting (LightGBM) ont obtenu les meilleures performances pour les deux tâches
- La classification d'espèces est globalement plus précise que la classification de maladies (F1 de 0.90 vs 0.81)
- L'écart de performance entre le meilleur modèle et les suivants est plus marqué pour la classification de maladies

### 4.2 Temps de Calcul
- LightGBM offre un bon compromis entre performance et temps d'entraînement
- Les modèles d'ensemble (Random Forest, Extra Trees) sont plus lents mais restent compétitifs

## 5. Recommandations

### 5.1 Modèles à Déployer
1. **Classification d'Espèces** : LightGBM
   - Raison : Meilleures performances globales
   - Fichier du modèle : `results/automl_especes/best_model_20250916_153446.pkl`

2. **Classification de Maladies** : LightGBM
   - Raison : Performance optimale et cohérente
   - Fichier du modèle : `results/automl_maladies/best_model_20250916_154829.pkl`

### 5.2 Considérations pour le Déploiement
- **Mémoire** : Les modèles LightGBM sont légers et rapides à l'inférence
- **Latence** : Temps d'inférence court, adapté aux applications en temps réel
- **Mise à l'échelle** : Supporte bien les grands volumes de données

## 6. Améliorations Possibles

1. **Traitement des Données**
   - Augmentation des données d'entraînement pour les classes sous-représentées
   - Techniques d'augmentation d'images plus sophistiquées

2. **Optimisation des Modèles**
   - Réglage fin des hyperparamètres des modèles sélectionnés
   - Exploration de l'embeddings des images avec des modèles pré-entraînés

3. **Architecture**
   - Mise en place d'un pipeline de prétraitement unifié
   - Implémentation d'un système de vote ou de stacking des meilleurs modèles

## 7. Conclusion

L'exploration a permis d'identifier LightGBM comme l'algorithme le plus performant pour les deux tâches de classification. Ces modèles offrent un excellent équilibre entre précision, rappel et temps d'exécution, les rendant idéaux pour un déploiement en production.

## 8. Fichiers Générés

- **Rapport de comparaison** : `reports/comparison_report_20250916_155157.md`
- **Graphiques de performance** : Dossier `figures/comparison/`
- **Modèles sauvegardés** : 
  - Espèces : `results/automl_especes/best_model_20250916_153446.pkl`
  - Maladies : `results/automl_maladies/best_model_20250916_154829.pkl`
