# Architecture 3: CNN mono-objectif - 1 tête unique (35 classes)

**Date**: 2025-11-07T16:30:42.547603

## Description

Modèle à 1 seule tête prédisant simultanément l'espèce ET l'état de santé/maladie.
Format des classes : `Espèce_État` (ex: `Apple_healthy`, `Tomato_late_blight`)

## Dataset

- **Nombre de classes**: 38
- **Format**: 14 espèces × (1 healthy + maladies) = 35 classes totales

### Comptes par classe

                                           class  count
                                Apple_apple_scab    630
                                 Apple_black_rot    621
                          Apple_cedar_apple_rust    275
                                   Apple_healthy   1645
                               Blueberry_healthy   1502
                 Cherry_(including_sour)_healthy    854
          Cherry_(including_sour)_powdery_mildew   1052
Corn_(maize)_cercospora_leaf_spot gray_leaf_spot    513
                        Corn_(maize)_common_rust   1192
                            Corn_(maize)_healthy   1162
               Corn_(maize)_northern_leaf_blight    985
                                 Grape_black_rot   1180
                      Grape_esca_(black_measles)   1383
                                   Grape_healthy    423
        Grape_leaf_blight_(isariopsis_leaf_spot)   1076
          Orange_haunglongbing_(citrus_greening)   5507
                            Peach_bacterial_spot   2297
                                   Peach_healthy    360
                     Pepper,_bell_bacterial_spot    997
                            Pepper,_bell_healthy   1478
                             Potato_early_blight   1000
                                  Potato_healthy    152
                              Potato_late_blight   1000
                               Raspberry_healthy    371
                                 Soybean_healthy   5090
                           Squash_powdery_mildew   1835
                              Strawberry_healthy    456
                          Strawberry_leaf_scorch   1109
                           Tomato_bacterial_spot   2127
                             Tomato_early_blight   1000
                                  Tomato_healthy   1591
                              Tomato_late_blight   1909
                                Tomato_leaf_mold    952
                       Tomato_septoria_leaf_spot   1771
     Tomato_spider_mites two-spotted_spider_mite   1676
                              Tomato_target_spot   1404
                      Tomato_tomato_mosaic_virus    373
            Tomato_tomato_yellow_leaf_curl_virus   5357

## Meilleures métriques (Validation/Test)

- **Val Accuracy**: 0.9959
- **Val Macro-F1**: 0.9943
- **Test Accuracy**: 0.9961
- **Test Macro-F1**: 0.9938

## Figures

- Courbes: `training_curves.png`
- Matrice de confusion: `confusion_matrix.png`
- Classification report: `classification_report.txt`
- Sanity grid: `sanity_check.png`

## Architecture

- **Backbone**: EfficientNetV2S (ImageNet pré-entraîné)
- **Stratégie**: Gel du backbone → Fine-tuning partiel
- **Sortie**: 1 Dense(35, softmax) - Classification multi-classe unique
- **Avantage**: Simplicité architecturale
- **Limite**: Perd la structure hiérarchique (espèce/maladie séparées)
