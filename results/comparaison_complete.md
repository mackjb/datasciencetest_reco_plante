# Comparaison Complète des Modèles

## 1. Avant Optimisation

### Reconnaissance des Espèces (14 classes)
| Modèle | F1 Score | Précision | Rappel |
|--------|----------|-----------|--------|
| XGBoost + LDA | 85.92% | 84.65% | 87.49% |
| XGBoost + PCA | 85.54% | 84.15% | 87.88% |
| XGBoost | 83.00% | 81.51% | 85.39% |

### Détection des Maladies (21 classes)
| Modèle | F1 Score | Précision | Rappel |
|--------|----------|-----------|--------|
| XGBoost + PCA | 79.81% | 78.53% | 82.54% |
| XGBoost + LDA | 75.90% | 74.00% | 79.76% |
| XGBoost | 74.59% | 73.71% | 77.43% |

## 2. Après Optimisation

### Reconnaissance des Espèces (14 classes)
| Modèle | F1 Score | Précision | Rappel |
|--------|----------|-----------|--------|
| XGBoost + PCA | 86.67% | - | - |
| XGBoost + LDA | 81.61% | - | - |
| XGBoost | 81.07% | - | - |

### Détection des Maladies (21 classes)
| Modèle | F1 Score | Précision | Rappel |
|--------|----------|-----------|--------|
| XGBoost + PCA | 73.59% | - | - |
| XGBoost | 48.87% | - | - |
| XGBoost + LDA | - | - | - |

## Analyse des Résultats

1. **Impact de l'optimisation** :
   - Légère amélioration pour XGBoost + PCA sur les espèces (+1.13%)
   - Baisse de performance pour les autres modèles, particulièrement marquée pour la détection des maladies

2. **Meilleurs modèles** :
   - Espèces : XGBoost + PCA (86.67%) après optimisation
   - Maladies : XGBoost + PCA (79.81%) avant optimisation

3. **Recommandations** :
   - Conserver XGBoost + PCA sans optimisation pour la détection des maladies
   - Utiliser la version optimisée de XGBoost + PCA pour la reconnaissance des espèces
   - Explorer pourquoi l'optimisation dégrade les performances pour certains modèles
