# Classification des maladies de plantes avec PlantVillage

Ce projet implémente un modèle de deep learning pour la classification des maladies de plantes en utilisant le dataset PlantVillage.

## Prérequis

- Python 3.8+
- pip
- Un environnement virtuel Python (recommandé)

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <votre-depot>
   cd datasciencetest_reco_plante
   ```

2. Créez et activez un environnement virtuel (optionnel mais recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Linux/Mac
   # OU
   .\venv\Scripts\activate  # Sur Windows
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Structure des dossiers

Assurez-vous d'avoir la structure suivante :
```
dataset/
└── plantvillage/
    ├── images/          # Dossier contenant toutes les images
    └── labels.csv       # Fichier CSV avec les labels
```

## Utilisation

### Entraînement du modèle
```bash
./run_plantvillage_classifier.sh train
```

### Évaluation du modèle
```bash
./run_plantvillage_classifier.sh evaluate
```

### Génération des explications (Grad-CAM, SHAP, LIME)
```bash
./run_plantvillage_classifier.sh explain
```

## Résultats

Les résultats seront enregistrés dans le dossier `results/deep_learning_plantvillage/` avec la structure suivante :
- `models/` : Modèles sauvegardés (.h5)
- `logs/` : Logs TensorBoard
- `plots/` : Graphiques et visualisations

## Personnalisation

Vous pouvez modifier les paramètres dans `scripts/config_plantvillage.py` pour :
- Changer la taille des images
- Ajuster la taille des lots (batch size)
- Modifier le nombre d'époques
- Changer le taux d'apprentissage
- etc.

## Aide

Pour afficher l'aide :
```bash
./run_plantvillage_classifier.sh
```
