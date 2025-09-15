# 🌿 Guide d'Utilisation du Modèle de Reconnaissance d'Espèces Végétales

## 📋 Fichiers Importants

1. **Modèle et Données**
   - `results/test_models/model_simple.json` : Modèle au format JSON (facile à lire)
   - `results/test_models/model_metadata.json` : Métadonnées du modèle
   - `results/test_models/example_usage.py` : Exemple de code pour utiliser le modèle

2. **Scripts Utiles**
   - `scripts/use_model.py` : Classe pour charger et utiliser le modèle
   - `scripts/test_with_sample.py` : Script de test avec une image exemple
   - `scripts/cleanup.py` : Pour nettoyer les fichiers temporaires

## 🚀 Comment Tester le Modèle

1. **Avec une image existante** :
   ```bash
   python scripts/test_with_sample.py chemin/vers/votre/image.jpg
   ```

2. **Avec l'image exemple** :
   - Placez votre image dans `data/examples/test_plant.jpg`
   - Exécutez simplement :
     ```bash
     python scripts/test_with_sample.py
     ```

## 🔍 Comment Utiliser le Modèle dans Votre Code

```python
from scripts.use_model import SimpleModel

# Initialiser le modèle
model = SimpleModel()

# Faire une prédiction sur une image
result = model.predict("chemin/vers/votre/image.jpg")

# Afficher les résultats
print(f"Espèce prédite: Classe {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2%}")
```

## 📊 Caractéristiques du Modèle

- Type : Random Forest
- Nombre d'arbres : 100
- Précision : ~70%
- Nombre de classes : 14
- Nombre de caractéristiques : 38

## 📋 Prochaines Étapes

1. Améliorer la précision du modèle
2. Ajouter plus de classes d'espèces
3. Créer une interface utilisateur
4. Déployer le modèle en production

## ❓ Besoin d'Aide ?

Consultez les commentaires dans les scripts ou exécutez :
```
python scripts/test_with_sample.py --help
```
