# Documentation des Données

Ce document décrit la structure et la gestion des données utilisées dans le projet.

## Sources de Données

### PlantVillage

- **Description** : Base de données d'images de plantes et de maladies
- **Lien** : [PlantVillage Dataset](https://plantvillage.psu.edu/)
- **Contenu** :
  - Plus de 54,000 images
  - 38 classes de plantes
  - 26 maladies différentes
- **Format** : Images JPG organisées par classe

### Flavia

- **Description** : Base de données de feuilles de plantes
- **Lien** : [Flavia Dataset](http://flavia.sourceforge.net/)
- **Contenu** :
  - 1,907 images
  - 32 espèces de plantes
- **Format** : Images JPG avec masques de segmentation

## Structure des Dossiers

```
data/
├── raw/                    # Données brutes (non modifiées)
│   ├── plantvillage/       # Données PlantVillage brutes
│   └── flavia/            # Données Flavia brutes
├── processed/             # Données traitées
│   ├── train/             # Données d'entraînement
│   │   ├── class1/        # Images de la classe 1
│   │   └── class2/        # Images de la classe 2
│   ├── val/               # Données de validation
│   └── test/              # Données de test
└── external/              # Données externes
```

## Préparation des Données

### Téléchargement

```bash
# Télécharger PlantVillage
python dataset/plantvillage/download_plantvillage.py

# Télécharger Flavia
python dataset/flavia/download_flavia.py
```

### Prétraitement

Les scripts de prétraitement effectuent les opérations suivantes :

1. Redimensionnement des images
2. Normalisation des couleurs
3. Augmentation des données (data augmentation)
4. Séparation train/val/test

```bash
# Exécuter le prétraitement
python scripts/preprocess.py --dataset plantvillage --output data/processed
```

## Format des Données

### Images

- **Format** : JPG
- **Taille** : 256x256 pixels (après prétraitement)
- **Espace colorimétrique** : RGB

### Annotations

Les annotations sont stockées au format COCO (JSON) avec les champs suivants :

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 256,
      "height": 256
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 2,
      "bbox": [x, y, width, height],
      "area": 1000,
      "segmentation": [...]
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "Tomato___healthy"
    }
  ]
}
```

## Bonnes Pratiques

1. Ne jamais modifier les fichiers dans `data/raw/`
2. Toujours utiliser des chemins relatifs
3. Documenter toute transformation appliquée aux données
4. Maintenir la séparation train/val/test

## Outils Recommandés

- **Visualisation** : LabelImg, CVAT
- **Traitement** : OpenCV, Albumentations
- **Gestion** : DVC (Data Version Control)
