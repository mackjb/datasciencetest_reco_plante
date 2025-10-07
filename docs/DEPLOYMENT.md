# Guide de Déploiement

Ce document explique comment déployer l'application en production.

## Prérequis

- Docker et Docker Compose
- Un serveur Linux (recommandé : Ubuntu 20.04+)
- Python 3.8+
- Au moins 4 Go de RAM

## Déploiement avec Docker (Recommandé)

1. Construire les images :
   ```bash
   docker-compose build
   ```

2. Démarrer les services :
   ```bash
   docker-compose up -d
   ```

3. Vérifier les logs :
   ```bash
   docker-compose logs -f
   ```

## Déploiement Manuel

1. Installer les dépendances système :
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv
   ```

2. Configurer l'environnement :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configurer les variables d'environnement :
   ```bash
   cp .env.example .env
   # Éditer le fichier .env selon vos besoins
   ```

4. Démarrer l'application :
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```

## Configuration Nginx (Optionnel)

Exemple de configuration Nginx pour servir l'application :

```nginx
server {
    listen 80;
    server_name votredomaine.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Mise à jour

Pour mettre à jour l'application :

```bash
git pull
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```
