# Comparaison des Performances par Modèle

## 1. XGBoost avec ACP

### 1.1 Maladies
| Maladie | Précision | Rappel | F1-score | Support |
|---------|-----------|--------|----------|---------|
| healthy | 99.73% | 99.87% | 99.8% | 3,013 |
| Common_rust | 97.06% | 97.06% | 97.06% | 238 |
| Haunglongbing | 95.27% | 96.91% | 96.09% | 1,102 |
| Tomato_Yellow_Leaf | 97.13% | 94.68% | 95.89% | 1,071 |
| Leaf_blight | 96.19% | 93.95% | 95.06% | 215 |
| Powdery_mildew | 93.43% | 91.16% | 92.28% | 577 |
| Leaf_scorch | 89.61% | 93.24% | 91.39% | 222 |
| Tomato_mosaic_virus | 87.65% | 94.67% | 91.03% | 75 |
| Bacterial_spot | 90.56% | 87.64% | 89.08% | 1,084 |
| Esca | 86.64% | 91.34% | 88.93% | 277 |
| Northern_Leaf_Blight | 86.15% | 85.71% | 85.93% | 196 |
| Spider_mites | 85.71% | 85.97% | 85.84% | 335 |
| Black_rot | 87.39% | 82.78% | 85.02% | 360 |
| Target_Spot | 79.02% | 80.43% | 79.72% | 281 |
| Cedar_apple_rust | 72.06% | 89.09% | 79.67% | 55 |
| Apple_scab | 76.09% | 83.33% | 79.55% | 126 |
| Septoria_leaf_spot | 80.76% | 78.25% | 79.48% | 354 |
| Cercospora_leaf_spot | 75.93% | 80.39% | 78.1% | 102 |
| Late_blight | 76.41% | 79.45% | 77.9% | 579 |
| Leaf_Mold | 78.49% | 76.84% | 77.66% | 190 |
| Early_blight | 74.38% | 74.94% | 74.66% | 399 |

### 1.2 Espèces
| Espèce | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Corn_(maize) | 98.04% | 97.53% | 97.78% | 769 |
| Blueberry | 94.53% | 98.0% | 96.24% | 300 |
| Squash | 95.68% | 96.46% | 96.07% | 367 |
| Orange | 96.0% | 95.83% | 95.91% | 1,102 |
| Soybean | 95.25% | 96.56% | 95.9% | 1,018 |
| Tomato | 95.66% | 94.18% | 94.92% | 3,627 |
| Grape | 94.53% | 93.72% | 94.12% | 812 |
| Peach | 93.97% | 93.97% | 93.97% | 531 |
| Raspberry | 94.52% | 93.24% | 93.88% | 74 |
| Strawberry | 89.85% | 93.29% | 91.54% | 313 |
| Cherry | 90.46% | 92.13% | 91.29% | 381 |
| Pepper_bell | 87.89% | 85.22% | 86.54% | 494 |
| Apple | 84.55% | 88.15% | 86.31% | 633 |
| Potato | 85.03% | 87.21% | 86.11% | 430 |

## 2. XGBoost avec LDA

### 2.1 Maladies
| Maladie | Précision | Rappel | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Tomato_Yellow_Leaf | 96.31% | 95.14% | 95.73% | 1,071 |
| Tomato_mosaic_virus | 92.31% | 96.0% | 94.12% | 75 |
| Common_rust | 96.05% | 92.02% | 93.99% | 238 |
| Haunglongbing | 92.21% | 94.56% | 93.37% | 1,102 |
| healthy | 92.57% | 85.23% | 88.75% | 3,013 |
| Powdery_mildew | 88.32% | 89.08% | 88.7% | 577 |
| Leaf_scorch | 86.64% | 90.54% | 88.55% | 222 |
| Leaf_blight | 88.21% | 86.98% | 87.59% | 215 |
| Esca | 81.76% | 87.36% | 84.47% | 277 |
| Northern_Leaf_Blight | 84.74% | 82.14% | 83.42% | 196 |
| Spider_mites | 79.72% | 85.67% | 82.59% | 335 |
| Bacterial_spot | 82.77% | 82.01% | 82.39% | 1,084 |
| Black_rot | 77.78% | 79.72% | 78.74% | 360 |
| Cercospora_leaf_spot | 71.07% | 84.31% | 77.13% | 102 |
| Septoria_leaf_spot | 72.15% | 76.84% | 74.42% | 354 |
| Target_Spot | 68.07% | 80.43% | 73.74% | 281 |
| Late_blight | 71.19% | 74.27% | 72.7% | 579 |
| Leaf_Mold | 71.73% | 72.11% | 71.92% | 190 |
| Early_blight | 70.42% | 72.18% | 71.29% | 399 |
| Apple_scab | 63.87% | 78.57% | 70.46% | 126 |
| Cedar_apple_rust | 50.63% | 72.73% | 59.7% | 55 |

### 2.2 Espèces
| Espèce | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Corn_(maize) | 96.79% | 98.05% | 97.42% | 769 |
| Blueberry | 95.35% | 95.67% | 95.51% | 300 |
| Orange | 92.41% | 95.01% | 93.69% | 1,102 |
| Soybean | 92.64% | 93.91% | 93.27% | 1,018 |
| Squash | 92.25% | 94.01% | 93.12% | 367 |
| Tomato | 96.03% | 90.05% | 92.94% | 3,627 |
| Grape | 90.96% | 92.98% | 91.96% | 812 |
| Raspberry | 94.2% | 87.84% | 90.91% | 74 |
| Strawberry | 91.26% | 90.1% | 90.68% | 313 |
| Peach | 86.55% | 89.64% | 88.07% | 531 |
| Potato | 78.62% | 87.21% | 82.69% | 430 |
| Cherry | 80.66% | 83.2% | 81.91% | 381 |
| Apple | 78.49% | 85.31% | 81.76% | 633 |
| Pepper_bell | 78.06% | 81.38% | 79.68% | 494 |

## 3. XGBoost de Base

### 3.1 Maladies
| Maladie | Précision | Rappel | F1-score | Support |
|---------|-----------|--------|----------|---------|
| healthy | 100.0% | 100.0% | 100.0% | 3,013 |
| Haunglongbing | 94.03% | 95.74% | 94.87% | 1,102 |
| Common_rust | 96.52% | 93.28% | 94.87% | 238 |
| Tomato_Yellow_Leaf | 95.67% | 92.81% | 94.22% | 1,071 |
| Powdery_mildew | 91.73% | 88.39% | 90.03% | 577 |
| Leaf_blight | 88.48% | 89.3% | 88.89% | 215 |
| Tomato_mosaic_virus | 81.61% | 94.67% | 87.65% | 75 |
| Bacterial_spot | 87.96% | 82.2% | 84.98% | 1,084 |

## 4. Synthèse des Performances

### 4.1 Moyennes par Modèle
| Modèle | Maladies (F1) | Espèces (F1) |
|--------|---------------|--------------|
| XGBoost + ACP | 85.4% | 93.8% |
| XGBoost + LDA | 83.2% | 90.2% |
| XGBoost seul | 90.39% | - |

### 4.2 Observations
- L'ACP offre de meilleures performances que la LDA pour ces données
- Les modèles sont très performants sur les classes majoritaires
- Les performances sont moins bonnes sur les classes rares et les maladies fongiques similaires
