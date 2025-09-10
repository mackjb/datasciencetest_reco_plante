# Comparaison des modèles de classification

| Modèle        | Type     |   Nb classes |   F1 moyen |   Précision |   Rappel |
|:--------------|:---------|-------------:|-----------:|------------:|---------:|
| XGBoost + LDA | Espèces  |           14 |      85.92 |       84.65 |    87.49 |
| XGBoost + PCA | Espèces  |           14 |      85.54 |       84.15 |    87.88 |
| XGBoost       | Espèces  |           14 |      83.00 |       81.51 |    85.39 |
| XGBoost + PCA | Maladies |           21 |      79.81 |       78.53 |    82.54 |
| XGBoost + LDA | Maladies |           21 |      75.90 |       74.00 |    79.76 |
| XGBoost       | Maladies |           21 |      74.59 |       73.71 |    77.43 |

## Meilleur modèle pour les Maladies: XGBoost + PCA
- Score F1: 79.81%
- Précision: 78.53%
- Rappel: 82.54%

## Meilleur modèle pour les Espèces: XGBoost + LDA
- Score F1: 85.92%
- Précision: 84.65%
- Rappel: 87.49%

