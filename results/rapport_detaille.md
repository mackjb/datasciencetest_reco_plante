# Rapport Détaillé des Performances des Modèles

## 1. Performances Globales

### 1.1 Maladies (21 classes)
| Métrique | Valeur |
|----------|---------|
| Score F1 moyen | 74.59% |
| Précision moyenne | 73.71% |
| Rappel moyen | 77.43% |

### 1.2 Espèces (14 classes)
| Métrique | Valeur |
|----------|---------|
| Score F1 moyen | 83.00% |
| Précision moyenne | 81.51% |
| Rappel moyen | 85.39% |

## 2. Détail par Classe - Modèle XGBoost avec ACP

### 2.1 Maladies (Top 5 et Bottom 5)
| Maladie | Précision | Rappel | F1-score | Support |
|---------|-----------|--------|----------|---------|
| healthy | 99.73% | 99.87% | 99.8% | 3,013 |
| Common_rust | 97.06% | 97.06% | 97.06% | 238 |
| Haunglongbing | 95.27% | 96.91% | 96.09% | 1,102 |
| Tomato_Yellow_Leaf | 97.13% | 94.68% | 95.89% | 1,071 |
| Leaf_blight | 96.19% | 93.95% | 95.06% | 215 |
| ... | ... | ... | ... | ... |
| Early_blight | 74.38% | 74.94% | 74.66% | 399 |
| Leaf_Mold | 78.49% | 76.84% | 77.66% | 190 |
| Late_blight | 76.41% | 79.45% | 77.9% | 579 |
| Cercospora_leaf_spot | 75.93% | 80.39% | 78.1% | 102 |
| Septoria_leaf_spot | 80.76% | 78.25% | 79.48% | 354 |

### 2.2 Espèces (Top 5 et Bottom 5)
| Espèce | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Corn_(maize) | 98.04% | 97.53% | 97.78% | 769 |
| Blueberry | 94.53% | 98.0% | 96.24% | 300 |
| Squash | 95.68% | 96.46% | 96.07% | 367 |
| Orange | 96.0% | 95.83% | 95.91% | 1,102 |
| Soybean | 95.25% | 96.56% | 95.9% | 1,018 |
| ... | ... | ... | ... | ... |
| Apple | 84.55% | 88.15% | 86.31% | 633 |
| Potato | 85.03% | 87.21% | 86.11% | 430 |
| Cherry | 90.46% | 92.13% | 91.29% | 381 |
| Strawberry | 89.85% | 93.29% | 91.54% | 313 |
| Raspberry | 94.52% | 93.24% | 93.88% | 74 |

## 3. Comparaison des Modèles

### 3.1 XGBoost avec ACP vs LDA

**Maladies**
- **ACP**: F1 moyen = 85.4%
- **LDA**: F1 moyen = 83.2%

**Espèces**
- **ACP**: F1 moyen = 93.8%
- **LDA**: F1 moyen = 90.2%

## 4. Analyse des Performances

### 4.1 Points Forts
- Excellentes performances sur les classes majoritaires (healthy, Tomato)
- Bon équilibre entre précision et rappel pour la plupart des classes
- Modèle robuste avec différentes techniques de réduction de dimension

### 4.2 Axes d'Amélioration
- Performances plus faibles sur les classes rares (ex: Cercospora_leaf_spot)
- Certaines confusions entre classes visuellement similaires

## 5. Recommandations

1. **Pour les classes sous-performantes** :
   - Augmenter le nombre d'échantillons d'entraînement pour les classes rares
   - Utiliser des techniques d'augmentation de données ciblées
   - Implémenter des poids de classe pour équilibrer l'apprentissage

2. **Amélioration du modèle** :
   - Tester d'autres techniques de réduction de dimension
   - Optimiser les hyperparamètres spécifiquement pour les classes problématiques
   - Explorer des architectures de modèles plus avancées (EfficientNet, Vision Transformer)

3. **Analyse approfondie** :
   - Examiner les exemples mal classés pour identifier les motifs d'erreur
   - Analyser les matrices de confusion pour détecter les confusions entre classes
   - Mettre en place une validation croisée plus poussée

## 6. Conclusion

Les modèles XGBoost montrent d'excellentes performances globales, particulièrement pour la classification des espèces végétales. L'ajout d'une étape de réduction de dimension avec ACP améliore significativement les performances par rapport à la LDA. Les prochaines étapes devraient se concentrer sur l'amélioration des performances pour les classes minoritaires et l'analyse des erreurs pour identifier les sources de confusion entre classes similaires.
