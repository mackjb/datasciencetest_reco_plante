# Classification des maladies de plantes avec PlantVillage

Ce projet implémente un modèle de deep learning pour la classification des maladies de plantes en utilisant le dataset PlantVillage. Le modèle est basé sur ResNet50 avec fine-tuning et inclut des fonctionnalités d'interprétabilité comme Grad-CAM et SHAP.

## Fonctionnalités

- Chargement efficace des données avec `ImageDataGenerator`
- Modèle ResNet50 avec fine-tuning des dernières couches
- Évaluation complète avec différentes métriques (précision, AUC, F1-score)
- Visualisation des explications avec Grad-CAM et SHAP
- Gestion des modèles (sauvegarde/chargement)
- Interface en ligne de commande facile à utiliser

## Prérequis

- Python 3.8+
- pip
- Un environnement virtuel Python (recommandé)
- Au moins 8 Go de RAM (16+ recommandé pour l'entraînement)
- Un GPU est fortement recommandé pour l'entraînement

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <votre-depot>
   cd datasciencetest_reco_plante
   ```

2. Créez et activez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Linux/Mac
   # OU
   .\venv\Scripts\activate  # Sur Windows
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements_plantvillage.txt
   ```

## Structure des dossiers

Le script s'attend à trouver les données dans la structure suivante :
```
dataset/
└── plantvillage/
    └── images/          # Dossier contenant les images classées par classe
        ├── classe1/     # Chaque sous-dossier représente une classe
        │   ├── img1.jpg
        │   └── ...
        ├── classe2/
        │   ├── img1.jpg
        │   └── ...
        └── ...
```

## Utilisation

Le script principal peut être exécuté avec différentes commandes :

### 1. Entraînement d'un nouveau modèle
```bash
# Avec augmentation des données (recommandé)
./run_plantvillage_classifier.sh train

# Sans augmentation des données
./run_plantvillage_classifier.sh train --no-augmentation
```

### 2. Évaluation d'un modèle existant
```bash
# Évalue le dernier modèle entraîné
./run_plantvillage_classifier.sh evaluate

# Évalue un modèle spécifique
./run_plantvillage_classifier.sh evaluate --model-dir results/plantvillage_20230901_123456
```

### 3. Génération d'explications
```bash
# Génère des explications avec le dernier modèle
./run_plantvillage_classifier.sh explain

# Génère des explications avec un modèle spécifique
./run_plantvillage_classifier.sh explain --model-dir results/plantvillage_20230901_123456
```

## Structure des résultats

Les résultats sont enregistrés dans le dossier `results/plantvillage_<timestamp>` avec la structure suivante :
```
results/
└── plantvillage_<timestamp>/
    ├── models/               # Modèles sauvegardés
    │   ├── best_model.h5     # Meilleur modèle (meilleure précision sur la validation)
    │   └── final_model.h5    # Dernier modèle entraîné
    ├── logs/                 # Logs TensorBoard
    ├── plots/                # Graphiques et visualisations
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── grad_cam_examples.png
    │   └── shap_explanations.png
    ├── evaluation_metrics.json  # Métriques d'évaluation
    └── config.json          # Configuration utilisée
```

## Personnalisation

Vous pouvez personnaliser les paramètres d'entraînement en modifiant la classe `Config` dans `scripts/plantvillage_classifier.py`. Les paramètres principaux incluent :

- Taille des images d'entrée
- Taille des lots (batch size)
- Nombre d'époques
- Taux d'apprentissage
- Taux de dropout
- Nombre de couches à dégeler pour le fine-tuning
- Paramètres d'augmentation des données

## Aide en ligne

Pour afficher l'aide complète :
```bash
./run_plantvillage_classifier.sh --help
```

## Résultats

Les résultats sont enregistrés dans le dossier `results/plantvillage_<timestamp>/` avec la structure décrite dans la section "Structure des résultats" ci-dessus.

## Exemple complet

Voici un exemple complet de workflow d'entraînement et d'évaluation :

```bash
# 1. Entraînement avec augmentation des données
./run_plantvillage_classifier.sh train

# 2. Évaluation du modèle entraîné
./run_plantvillage_classifier.sh evaluate

# 3. Génération d'explications
./run_plantvillage_classifier.sh explain
```

## Dépannage

### Problèmes courants

1. **Mémoire insuffisante** :
   - Réduisez la taille du batch dans la configuration
   - Utilisez des images plus petites
   - Activez le garbage collection de Python

2. **Erreurs de chargement des données** :
   - Vérifiez la structure des dossiers
   - Assurez-vous que les images sont dans des sous-dossiers par classe
   - Vérifiez les permissions des fichiers

3. **Problèmes de performance** :
   - Activez l'accélération GPU si disponible
   - Utilisez un nombre de workers approprié pour le chargement des données
   - Vérifiez que votre système dispose de suffisamment de RAM

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à soumettre des issues ou des pull requests pour des améliorations ou des corrections de bugs.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
