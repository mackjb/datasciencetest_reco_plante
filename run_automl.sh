#!/bin/bash

# Script de démarrage du pipeline AutoML

echo "========================================"
echo "  LANCEMENT DU PIPELINE AUTOML"
echo "  (Espèces et Maladies)"
echo "========================================"

# Vérification des dépendances
echo "\n🔍 Vérification des dépendances..."
pip install -q pycaret optuna scikit-learn pandas numpy matplotlib seaborn

# Création des dossiers nécessaires
echo "📂 Création des dossiers de sortie..."
mkdir -p results/automl

# Fonction pour exécuter avec une cible spécifique
run_for_target() {
    local target_type=$1
    echo "\n🚀 Démarrage du pipeline pour : $target_type"
    
    # Mise à jour de la configuration
    python -c "
import json
config_path = 'config/automl_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
config['data']['target_type'] = '$target_type'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
    "
    # Exécution du pipeline
    python scripts/automl_pipeline.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Pipeline pour $target_type terminé avec succès !"
    else
        echo "❌ Erreur lors de l'exécution pour $target_type"
        return 1
    fi
}

# Exécution pour les espèces
echo "\n🌿 DÉMARRAGE DE L'ANALYSE DES ESPÈCES"
echo "----------------------------------------"
run_for_target "espece"

# Exécution pour les maladies
echo "\n🦠 DÉMARRAGE DE L'ANALYSE DES MALADIES"
echo "----------------------------------------"
run_for_target "maladie"

# Message final
echo "\n========================================"
echo "  ANALYSE TERMINÉE"
echo "========================================"
echo "📁 Résultats disponibles dans : results/automl/"
echo "📋 Logs disponibles dans : automl_pipeline.log"
echo ""
