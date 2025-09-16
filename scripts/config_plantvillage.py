"""
Configuration pour le classifieur PlantVillage

Ce fichier contient les chemins et paramètres pour le classifieur PlantVillage.
"""
from pathlib import Path

# Chemins des données
DATA_DIR = Path("dataset/plantvillage/")
IMAGE_DIR = DATA_DIR / "images"
LABEL_CSV = DATA_DIR / "labels.csv"

# Paramètres du modèle
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LAYERS = 50

# Dossiers de sortie
OUTPUT_DIR = Path("results/deep_learning_plantvillage")
MODEL_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Création des dossiers si nécessaire
for directory in [MODEL_DIR, LOGS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Classes (sera rempli dynamiquement)
CLASS_NAMES = []
