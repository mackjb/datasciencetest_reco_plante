# Lancer l'application Streamlit

## Prérequis

L'environnement conda doit être activé avec toutes les dépendances installées :

```bash
conda env create -f conda_env.yml
conda activate conda_env
```

## Lancement

Depuis la **racine du projet**, exécuter :

```bash
streamlit run Streamlit/app.py
```

L'application sera accessible à l'adresse : **http://localhost:8501**

## Structure de l'application

| Page | Description |
|------|-------------|
| **Le projet** | Présentation du projet et de l'équipe |
| **Les jeux de données** | Analyse exploratoire et preprocessing |
| **Méthodologie ML-DL** | Approche méthodologique |
| **Machine Learning** | Résultats des modèles ML classiques |
| **Deep Learning** | Exploration des 9 architectures DL |
| **PoCs** | Démonstrations interactives avec Grad-CAM |
| **Conclusion & Perspectives** | Synthèse et perspectives |

## Dépendances principales

- `streamlit`
- `pandas`
- `plotly`
