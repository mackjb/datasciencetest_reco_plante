# Guide des Modèles

Ce document décrit les différents modèles disponibles dans le projet et comment les utiliser.

## 📚 Table des matières

- [Modèles Disponibles](#-modèles-disponibles)
- [AutoML](#-automl)
- [PlantVillage](#-plantvillage)
- [Utilisation des Modèles](#-utilisation-des-modèles)
- [Entraînement Personnalisé](#-entraînement-personnalisé)

## 🤖 Modèles Disponibles

### YOLOv8

Pour la classification d'images de plantes et de maladies.

**Fonctionnalités** :
- Entraînement de modèles de classification
- Prédiction sur de nouvelles images
- Export vers ONNX pour le déploiement

Voir la [documentation YOLOv8](models/yolov8/README.md) pour plus de détails.

### AutoML

Pour l'optimisation automatique des modèles de machine learning.

**Fonctionnalités** :
- Recherche d'hyperparamètres automatique
- Comparaison de modèles
- Optimisation des métriques de performance

### ResNet50 (PlantVillage)

Modèle de deep learning pour la classification des maladies de plantes.

**Fonctionnalités** :
- Fine-tuning de ResNet50
- Explications des prédictions avec Grad-CAM et SHAP
- Évaluation complète des performances

## ⚙️ AutoML

### Prérequis

- Python 3.7+
- pip

### Utilisation

```bash
# Rendre le script exécutable (une seule fois)
chmod +x run_automl.sh

# Lancer le pipeline
./run_automl.sh
```

### Structure des Fichiers

```
models/automl/
├── train/                   # Scripts d'entraînement
│   ├── automl_pipeline.py   # Pipeline principal
│   └── run_simple_automl.py # Point d'entrée
├── eval/                   # Évaluation des modèles
│   └── compare_models.py   # Comparaison des performances
└── config/                 # Configurations
    └── automl_config.json  # Paramètres AutoML
```

## 🌱 PlantVillage

### Fonctionnalités

- Chargement efficace des données avec `ImageDataGenerator`
- Modèle ResNet50 avec fine-tuning
- Visualisation des explications (Grad-CAM, SHAP)
- Interface en ligne de commande

### Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset
python dataset/plantvillage/download_plantvillage.py
```

### Entraînement

```bash
python models/plantvillage/train.py \
    --data_dir data/plantvillage \
    --output_dir results/plantvillage \
    --epochs 20 \
    --batch_size 32
```

## 🚀 Utilisation des Modèles

### Fichiers Importants

- `results/test_models/model_simple.json` - Modèle sérialisé
- `results/test_models/model_metadata.json` - Métadonnées du modèle
- `results/test_models/example_usage.py` - Exemple d'utilisation

### Tester avec une Image

```bash
python scripts/test_with_sample.py chemin/vers/image.jpg
```

### Utilisation dans le Code

```python
from scripts.use_model import PlantClassifier

# Charger le modèle
classifier = PlantClassifier('results/test_models/model_simple.json')

# Faire une prédiction
prediction = classifier.predict('chemin/vers/image.jpg')
print(f"Classe prédite: {prediction['class']}")
print(f"Confiance: {prediction['confidence']:.2%}")
```

## 🎓 Entraînement Personnalisé

### Préparer les Données

Organisez vos données comme suit :

```
data/
  train/
    classe1/
      image1.jpg
      image2.jpg
    classe2/
      image3.jpg
  val/
    classe1/
      image4.jpg
    classe2/
      image5.jpg
```

### Lancer l'Entraînement

```bash
python scripts/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --output_dir results/my_model
```

## 🔧 Dépannage

### Problèmes Courants

1. **Erreurs de mémoire** :
   - Réduisez la taille du batch
   - Utilisez des images plus petites
   - Activez le mixed precision training

2. **Erreurs de chargement des données** :
   - Vérifiez la structure des dossiers
   - Assurez-vous que les images sont valides
   - Vérifiez les permissions

3. **Performances médiocres** :
   - Augmentez la taille du jeu d'entraînement
   - Essayez l'augmentation de données
   - Ajustez les hyperparamètres
