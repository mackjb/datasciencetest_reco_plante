# Guide de Développement

Ce document fournit des instructions pour les développeurs souhaitant contribuer au projet.

## Structure du Projet

```
datasciencetest_reco_plante/
├── dataset/               # Données brutes (non versionnées)
├── data/                  # Données traitées (train/val/test)
├── models/                # Modèles et entraînements
│   ├── yolov8/            # Modèle YOLOv8
│   └── automl/            # Modèles AutoML
├── api/                   # API FastAPI
├── scripts/               # Scripts utilitaires
├── tests/                 # Tests unitaires et d'intégration
├── configs/               # Fichiers de configuration
└── docs/                  # Documentation
```

## Configuration de l'environnement

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/plant-disease-classifier.git
   cd plant-disease-classifier
   ```

2. Créer un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   .\venv\Scripts\activate  # Windows
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Tests

Pour exécuter les tests :

```bash
pytest tests/
```

## Contribution

1. Créer une branche pour votre fonctionnalité :
   ```bash
   git checkout -b feature/nouvelle-fonctionnalite
   ```

2. Faire vos modifications
3. Tester vos changements
4. Soumettre une Pull Request

## Bonnes pratiques

- Suivre le style de code PEP 8
- Écrire des tests pour les nouvelles fonctionnalités
- Documenter le code avec des docstrings
- Mettre à jour la documentation si nécessaire
