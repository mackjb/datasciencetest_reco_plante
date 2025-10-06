# Plant Disease Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)

Un outil de classification des maladies des plantes utilisant le Deep Learning, basé sur le dataset PlantVillage.

## Fonctionnalités

- **Classification d'images** de plantes avec leur état de santé
- **Explications des prédictions** avec Grad-CAM, SHAP et LIME
- **Séparation stricte** des données d'entraînement, de validation et de test
- **Gestion de la mémoire** optimisée pour les opérations intensives
- **Logs détaillés** et suivi des expériences

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/plant-disease-classifier.git
   cd plant-disease-classifier
   ```

2. Créer un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Linux/Mac
   # OU
   .\venv\Scripts\activate  # Sur Windows
   ```
## Structure du projet

datasciencetest_reco_plante/
├── dataset/                         # Données brutes (non versionnées)
├── data/                            # Données dérivées (split train/valid, features)
├── results/                         # Sorties et logs
├── models/
│   ├── yolov8/
│   │   ├── train/                   # Entraînement YOLOv8
│   │   │   └── yolov8_train.py
│   │   ├── predict/                 # Inférence YOLOv8
│   │   │   └── predict_yolov8.py
│   │   ├── export/                  # Export YOLOv8 (ONNX)
│   │   │   └── export_yolov8_onnx.py
│   │   └── eval/                    # Évaluations/plots (si besoin)
│   ├── automl/
│   │   ├── train/
│   │   │   ├── automl_pipeline.py
│   │   │   ├── run_simple_automl.py
│   │   │   └── save_model_simple.py
│   │   ├── eval/
│   │   │   ├── compare_automl_results.py
│   │   │   ├── list_models.py
│   │   │   ├── plot_actual_models.py
│   │   │   ├── plot_learning_curves_comparison.py
│   │   │   ├── plot_model_comparison.py
│   │   │   ├── visualize_global_results.py
│   │   │   └── visualize_results.py
│   │   └── report/
│   │       ├── create_detailed_scores.py
│   │       ├── generate_detailed_scores_report.py
│   │       ├── generate_results_csv.py
│   │       ├── generate_results_table.py
│   │       └── generate_species_report.py
│   └── xgboost/
│       └── train/
│           └── finetune_xgboost.py
├── scripts/                        # Utilitaires généraux (ex: utils.py)
├── archive/                        # Poubelle/archives ignorées par git
├── README.md
├── Makefile
└── requirements.txt

 ## Makefile commands
 
 ### YOLOv8
 
 - `make train-yolo`: Entraîne le modèle YOLOv8.
 - `make predict-yolo INPUT=/chemin/vers/image_ou_dossier [TOPK=5]`: Inférence YOLOv8.
 - `make export-yolo-onnx`: Exporte le modèle YOLOv8 en ONNX.

### AutoML

 - `make automl-train`: Lance le pipeline AutoML.
 - `make automl-eval`: Génère les comparatifs et évaluations AutoML.

### XGBoost

 - `make xgb-train`: Entraîne/affine le modèle XGBoost.

## Préparation des données

Placez votre dataset PlantVillage (ou similaire) organisé par classes:

```
dataset/plantvillage/images/
├── ClasseA/
│   ├── img001.jpg
│   └── ...
├── ClasseB/
│   ├── img101.jpg
│   └── ...
└── ...
```

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Auteur

Votre Nom - votre.email@example.com

---

## YOLOv8 - Classification PlantVillage

Cette section décrit le pipeline YOLOv8 (entraînement, évaluation, inférence et export ONNX) organisé sous `models/yolov8/`.

- **Entraînement**: `models/yolov8/train/yolov8_train.py`
- **Inférence**: `models/yolov8/predict/predict_yolov8.py`
- **Export ONNX**: `models/yolov8/export/export_yolov8_onnx.py`

### Pré-requis spécifiques

En complément des dépendances existantes, installez:

```bash
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tqdm scikit-learn pandas matplotlib pillow
```

Remarque: pour GPU CUDA, suivez le guide d'installation PyTorch: https://pytorch.org/get-started/locally/

### Chemins utilisés

- **Dataset (fixe)**:
  - `/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented`
  - Attendu: un dossier par classe contenant des images (`.jpg/.jpeg/.png`).
- **Données traitées (split train/valid)**:
  - `/workspaces/datasciencetest_reco_plante/data/PlantVillage_Processed`
- **Résultats**:
  - `/workspaces/datasciencetest_reco_plante/results/yolov8_segmented_finetune/`

### Lancer l'entraînement

```bash
make train-yolo
```

Sorties dans `results/yolov8_segmented_finetune/`:
- `results.csv`, `weights/best.pt`
- `classification_report.csv`, `predictions_probs.csv`
- Figures: `loss_curves.png`, `loss_acc_curves.png`, `overfit_gap.png`, `confusion_matrix.png`
- Logs: `train.log`

Le split train/valid est réutilisé s'il existe déjà. Reproductibilité activée (seed=42).

### Inférence (image ou dossier)

```bash
# Image
make predict-yolo INPUT=/chemin/vers/image.jpg

# Dossier
make predict-yolo INPUT=/chemin/vers/dossier_images TOPK=5
```

- Poids par défaut: `results/yolov8_segmented_finetune/weights/best.pt`
- `--save` crée des copies simples sous `predictions/` (classification).

### Export ONNX

```bash
make export-yolo-onnx
```

Le fichier `model.onnx` est généré dans `--outdir`.
