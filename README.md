# 🌿 Plant Disease Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)

Un outil avancé de classification des maladies des plantes utilisant le Deep Learning, basé sur les datasets PlantVillage et Flavia.

## 🌟 Fonctionnalités

- **Classification d'images** de plantes avec détection de maladies
- **Modèles supportés** :
  - YOLOv8 pour la classification
  - Modèles AutoML pour l'optimisation automatique
- **Explications des prédictions** avec Grad-CAM, SHAP et LIME
- **API RESTful** pour l'intégration facile
- **Déploiement conteneurisé** avec Docker

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.8+
- pip
- (Optionnel) Docker pour le déploiement

### Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/plant-disease-classifier.git
   cd plant-disease-classifier
   ```

2. Créer et activer un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OU
   .\venv\Scripts\activate  # Windows
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Structure du Projet

```
plant-disease-classifier/
├── api/                   # API FastAPI
│   ├── app.py            # Point d'entrée de l'API
│   ├── models/           # Modèles pour l'API
│   └── routes/           # Routes de l'API
├── configs/              # Fichiers de configuration
│   └── default.yaml      # Configuration principale
├── data/                 # Données traitées (non versionnées)
│   ├── train/            # Données d'entraînement
│   ├── val/              # Données de validation
│   └── test/             # Données de test
├── dataset/              # Scripts de téléchargement
│   ├── plantvillage/     # Téléchargement PlantVillage
│   └── flavia/           # Téléchargement Flavia
├── docs/                 # Documentation
│   ├── API.md           # Documentation de l'API
│   ├── DATA.md          # Documentation des données
│   ├── DEPLOYMENT.md    # Guide de déploiement
│   └── DEVELOPMENT.md   # Guide de développement
├── models/               # Modèles et entraînements
│   ├── yolov8/          # Modèle YOLOv8
│   └── automl/          # Modèles AutoML
├── notebooks/            # Notebooks Jupyter
├── scripts/             # Scripts utilitaires
│   ├── preprocess.py    # Prétraitement des données
│   └── train.py         # Script d'entraînement
├── tests/               # Tests unitaires
├── .gitignore
├── docker-compose.yml   # Configuration Docker
├── Dockerfile           # Fichier de build Docker
└── requirements.txt     # Dépendances Python
```

## 📚 Documentation Complète

- [Guide de Développement](docs/DEVELOPMENT.md) - Comment contribuer au projet
- [Guide de Déploiement](docs/DEPLOYMENT.md) - Comment déployer en production
- [Documentation de l'API](docs/API.md) - Documentation complète des endpoints
- [Documentation des Données](docs/DATA.md) - Structure et gestion des données
- [Guide des Modèles](docs/MODELS.md) - Documentation des modèles disponibles et leur utilisation
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
