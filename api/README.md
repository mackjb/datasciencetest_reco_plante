# API de Reconnaissance de Plantes et Maladies

Cette API expose les modèles de classification pour la reconnaissance d'espèces de plantes et de maladies.

## Prérequis

- Python 3.8+
- pip
- Docker (optionnel, pour le déploiement en conteneur)

## Installation

1. Cloner le dépôt
2. Créer un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### Lancer l'API en mode développement

```bash
uvicorn app:app --reload
```

L'API sera disponible à l'adresse : http://localhost:8000

### Tester l'API

Un script de test est disponible :

```bash
python test_api.py
```

### Déploiement avec Docker

1. Construire l'image Docker :
   ```bash
   docker build -t reco-plante-api .
   ```

2. Lancer le conteneur :
   ```bash
   docker run -d --name reco-plante-api -p 8000:8000 reco-plante-api
   ```

## Endpoints

- `GET /` : Page d'accueil de l'API
- `GET /model_info/{task}` : Informations sur le modèle (especes ou maladies)

## Documentation

La documentation interactive est disponible à l'adresse :
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)
