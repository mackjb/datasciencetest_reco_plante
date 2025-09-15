# Pipeline AutoML pour la Classification de Plantes

Ce pipeline permet d'entraîner et d'optimiser automatiquement des modèles de classification pour la reconnaissance d'espèces et de maladies de plantes.

## 🚀 Démarrage Rapide

1. **Prérequis** :
   - Python 3.7+
   - pip

2. **Lancement** :
   ```bash
   # Rendre le script exécutable (une seule fois)
   chmod +x run_automl.sh
   
   # Lancer le pipeline
   ./run_automl.sh
   ```

## 📂 Structure des Fichiers

```
.
├── config/
│   └── automl_config.json    # Configuration du pipeline
├── data/                     # Dossier pour les données
│   └── processed/
├── results/                  # Résultats et modèles
│   └── automl/
├── scripts/
│   └── automl_pipeline.py    # Script principal
├── run_automl.sh             # Script de démarrage
└── README_AUTOML.md          # Ce fichier
```

## ⚙️ Configuration

Modifiez `config/automl_config.json` pour personnaliser :

```json
{
    "data": {
        "csv_path": "chemin/vers/vos/donnees.csv",
        "target_type": "espece",  # ou "maladie"
        "test_size": 0.2
    },
    "preprocessing": {
        "normalize": true,
        "feature_selection": true,
        "fix_imbalance": true
    },
    "models": {
        "include": ["xgboost", "lightgbm", "catboost"],
        "optimize_metric": "F1"
    }
}
```

## 📊 Résultats

Les résultats sont sauvegardés dans `results/automl/` :
- Modèles entraînés
- Métriques de performance
- Visualisations
- Fichiers de prédictions

## 🔍 Journalisation

Tous les logs sont enregistrés dans `automl_pipeline.log`

## 📝 Notes

- Le pipeline utilise PyCaret pour l'automatisation du machine learning
- L'optimisation est effectuée avec Optuna
- Les modèles sont sauvegardés au format PKL
