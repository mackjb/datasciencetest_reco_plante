# Documentation de l'API

Cette documentation décrit les endpoints de l'API de reconnaissance de plantes et maladies.

## Base URL

```
https://api.votredomaine.com/v1
```

## Authentification

Toutes les requêtes nécessitent un token d'API :

```
Authorization: Bearer VOTRE_TOKEN_API
```

## Endpoints

### Classifier une image

Classifie une image de plante et retourne les prédictions.

```http
POST /classify
```

**Paramètres :**

- `image` (obligatoire) : Fichier image à classifier (JPG, PNG)
- `threshold` (optionnel, défaut=0.5) : Seuil de confiance minimum (0-1)

**Exemple de requête :**

```bash
curl -X POST \
  -H "Authorization: Bearer VOTRE_TOKEN" \
  -F "image=@chemin/vers/image.jpg" \
  https://api.votredomaine.com/v1/classify
```

**Réponse réussie (200 OK) :**

```json
{
  "success": true,
  "predictions": [
    {
      "class": "Tomato___healthy",
      "confidence": 0.97,
      "disease": false
    },
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.02,
      "disease": true
    }
  ]
}
```

### Obtenir l'historique des prédictions

Retourne l'historique des prédictions de l'utilisateur.

```http
GET /history
```

**Paramètres de requête :**

- `limit` (optionnel, défaut=10) : Nombre maximum de résultats à retourner
- `offset` (optionnel, défaut=0) : Décalage pour la pagination

**Exemple de réponse :**

```json
{
  "success": true,
  "history": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "timestamp": "2023-06-15T10:30:00Z",
      "image_url": "/uploads/123e4567.jpg",
      "predictions": [
        {
          "class": "Apple___healthy",
          "confidence": 0.95,
          "disease": false
        }
      ]
    }
  ],
  "total": 42
}
```

## Codes d'erreur

| Code | Description |
|------|-------------|
| 400 | Requête invalide |
| 401 | Non autorisé |
| 404 | Ressource non trouvée |
| 429 | Trop de requêtes |
| 500 | Erreur serveur |
