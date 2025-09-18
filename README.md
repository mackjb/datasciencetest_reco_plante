# Plant Disease Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)

Un outil de classification des maladies des plantes utilisant le Deep Learning, basé sur le dataset PlantVillage.

## Fonctionnalités

- **Classification d'images** de plantes avec leur état de santé
- **Modèle basé sur ResNet50** avec fine-tuning
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

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Structure du projet

```
plant-disease-classifier/
├── dataset/                    # Dossier pour les données (à créer)
│   └── plantvillage/
│       └── images/             # Images classées par dossier de classe
├── results/                    # Dossier de sortie (créé automatiquement)
│   └── plantvillage_<timestamp>/
│       ├── models/             # Modèles sauvegardés
│       ├── logs/               # Logs TensorBoard
│       └── plots/              # Graphiques et visualisations
├── scripts/                    # Code source
│   ├── __init__.py
│   ├── plantvillage_classifier.py  # Classes principales
│   └── utils.py               # Fonctions utilitaires
├── run_pipeline.py            # Script principal
├── setup.py                   # Configuration du package
└── requirements.txt           # Dépendances Python
```

## Utilisation

### Préparation des données

1. Téléchargez le dataset PlantVillage et placez-le dans `dataset/plantvillage/images/`
2. Structure attendue :
   ```
   dataset/plantvillage/images/
   ├── class1/
   │   ├── image1.jpg
   │   └── ...
   ├── class2/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

### Entraînement et évaluation

```bash
python run_pipeline.py \
  --data_dir dataset/plantvillage/images \
  --batch_size 32 \
  --epochs 20 \
  --learning_rate 1e-4 \
  --fine_tune_lr 1e-5 \
  --output_dir results
```

### Options disponibles

- `--data_dir`: Chemin vers le dossier contenant les images classées par classe (défaut: `dataset/plantvillage/images`)
- `--batch_size`: Taille des lots pour l'entraînement (défaut: 32)
- `--epochs`: Nombre d'époques d'entraînement (défaut: 20)
- `--learning_rate`: Taux d'apprentissage initial (défaut: 1e-4)
- `--fine_tune_lr`: Taux d'apprentissage pour le fine-tuning (défaut: 1e-5)
- `--output_dir`: Dossier de sortie pour les résultats (défaut: `results`)

## Résultats

Les résultats sont enregistrés dans le dossier `results/plantvillage_<timestamp>/` avec :

- `models/`: Modèles sauvegardés au format HDF5
- `logs/`: Fichiers pour la visualisation avec TensorBoard
- `plots/`: Graphiques de performance et explications
- `metrics.json`: Métriques d'évaluation détaillées
- `pipeline.log`: Logs d'exécution

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Auteur

Votre Nom - votre.email@example.com
