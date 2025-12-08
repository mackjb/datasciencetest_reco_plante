# Architecture 3: CNN mono-objectif - 2 têtes | Task: disease_all
Date: 2025-10-27T14:33:48.629181

## Dataset
- Classes: 21
- Liste (triée):
  - apple_scab
  - bacterial_spot
  - black_rot
  - cedar_apple_rust
  - cercospora_leaf_spot gray_leaf_spot
  - common_rust
  - early_blight
  - esca_(black_measles)
  - haunglongbing_(citrus_greening)
  - healthy
  - late_blight
  - leaf_blight_(isariopsis_leaf_spot)
  - leaf_mold
  - leaf_scorch
  - northern_leaf_blight
  - powdery_mildew
  - septoria_leaf_spot
  - spider_mites two-spotted_spider_mite
  - target_spot
  - tomato_mosaic_virus
  - tomato_yellow_leaf_curl_virus

### Comptes par classe
                               class  count
                          apple_scab    630
                      bacterial_spot   5421
                           black_rot   1801
                    cedar_apple_rust    275
 cercospora_leaf_spot gray_leaf_spot    513
                         common_rust   1192
                        early_blight   2000
                esca_(black_measles)   1383
     haunglongbing_(citrus_greening)   5507
                             healthy  15084
                         late_blight   2909
  leaf_blight_(isariopsis_leaf_spot)   1076
                           leaf_mold    952
                         leaf_scorch   1109
                northern_leaf_blight    985
                      powdery_mildew   2887
                  septoria_leaf_spot   1771
spider_mites two-spotted_spider_mite   1676
                         target_spot   1404
                 tomato_mosaic_virus    373
       tomato_yellow_leaf_curl_virus   5357

## Meilleures métriques (Validation/Test)
- Val Accuracy: 0.9963
- Val Macro-F1: 0.9922
- Test Accuracy: 0.9964
- Test Macro-F1: 0.9924

## Figures
- Courbes: training_curves.png
- Matrice de confusion: confusion_matrix.png
- Classification report: classification_report.txt
- Sanity grid: sanity_check.png
