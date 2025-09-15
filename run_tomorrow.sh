#!/bin/bash

# Script pour faciliter le démarrage demain
# Auteur: Cascade AI Assistant
# Date: $(date +%Y-%m-%d)

echo "========================================"
echo "  BIENVENUE DANS LE PROJET RECO PLANTE  "
echo "========================================"
echo ""

# Vérifier l'environnement
if [ ! -d "venv" ]; then
    echo "🌱 Création de l'environnement virtuel..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ Environnement virtuel détecté"
    source venv/bin/activate
fi

# Nettoyer l'espace de travail
echo "🧹 Nettoyage de l'espace de travail..."
python scripts/cleanup.py

# Afficher les modèles disponibles
echo ""
echo "🌿 MODÈLES DISPONIBLES"
echo "----------------------"
ls -la results/test_models/

# Afficher les métadonnées du modèle
echo ""
echo "📊 MÉTADONNÉES DU MODÈLE"
echo "------------------------"
if [ -f "results/test_models/model_metadata.json" ]; then
    cat results/test_models/model_metadata.json | python -m json.tool
else
    echo "Aucun fichier de métadonnées trouvé."
fi

# Instructions pour l'utilisation
echo ""
echo "🚀 POUR COMMENCER"
echo "----------------"
echo "1. Placez vos images à analyser dans le dossier 'data/predict/'
echo "2. Exécutez: python scripts/predict_species.py --input data/predict/votre_image.jpg
"
# Activer l'environnement par défaut
echo "Pour activer l'environnement virtuel, exécutez:"
echo "source venv/bin/activate"
echo ""
echo "Bonne session ! 🌱"
echo ""
