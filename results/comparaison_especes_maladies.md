# Comparaison des Modèles (Avant Optimisation)

## 1. Reconnaissance des Espèces (14 classes)

| Modèle | F1 Score | Précision | Rappel |
|--------|----------|-----------|--------|
| XGBoost + LDA | 85.92% | 84.65% | 87.49% |
| XGBoost + PCA | 85.54% | 84.15% | 87.88% |
| XGBoost | 83.00% | 81.51% | 85.39% |

## 2. Détection des Maladies (21 classes)

| Modèle | F1 Score | Précision | Rappel |
|--------|----------|-----------|--------|
| XGBoost + PCA | 79.81% | 78.53% | 82.54% |
| XGBoost + LDA | 75.90% | 74.00% | 79.76% |
| XGBoost | 74.59% | 73.71% | 77.43% |

## Observations Clés

1. **Meilleur modèle global** :
   - Espèces : XGBoost + LDA (F1: 85.92%)
   - Maladies : XGBoost + PCA (F1: 79.81%)

2. **Performance par métrique** :
   - Les rappels sont systématiquement plus élevés que les précisions
   - L'écart entre précision et rappel est plus marqué pour les maladies

3. **Comparaison des approches** :
   - LDA > PCA pour les espèces
   - PCA > LDA pour les maladies
   - L'approche de base (sans PCA/LDA) est la moins performante dans les deux cas
