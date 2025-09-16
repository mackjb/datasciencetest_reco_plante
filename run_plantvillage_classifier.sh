#!/bin/bash

# Script de démarrage pour le classifieur PlantVillage
# Utilisation: ./run_plantvillage_classifier.sh [train|evaluate|explain] [options]

# Fonction d'aide
show_help() {
    echo "Utilisation: $0 [train|evaluate|explain] [options]"
    echo ""
    echo "Commandes:"
    echo "  train       Entraîne un nouveau modèle"
    echo "  evaluate    Évalue un modèle existant"
    echo "  explain     Génère des explications avec un modèle existant"
    echo ""
    echo "Options:"
    echo "  --model-dir DIR   Chemin vers le répertoire du modèle (pour evaluate/explain)"
    echo "  --no-augmentation Désactive l'augmentation des données (pour l'entraînement)"
    echo "  --help            Affiche ce message d'aide"
    echo ""
    echo "Exemples:"
    echo "  $0 train"
    echo "  $0 evaluate --model-dir results/plantvillage_20230901_123456"
    echo "  $0 explain --model-dir results/plantvillage_20230901_123456"
}

# Vérification des arguments
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Vérification de la commande
COMMAND=""
MODEL_DIR=""
NO_AUGMENTATION=false

# Traitement des arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        train|evaluate|explain)
            COMMAND="$1"
            shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --no-augmentation)
            NO_AUGMENTATION=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Erreur: Argument inconnu: $1"
            show_help
            exit 1
            ;;
    esac
done

# Vérification de la commande
if [ -z "$COMMAND" ]; then
    echo "Erreur: Aucune commande spécifiée"
    show_help
    exit 1
fi

# Vérification de Python et des dépendances
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python 3 n'est pas installé"
    exit 1
fi

# Installation des dépendances si nécessaire
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements_plantvillage.txt
else
    source venv/bin/activate
fi

# Exécution de la commande
case "$COMMAND" in
    train)
        echo "Démarrage de l'entraînement..."
        python -m scripts.plantvillage_classifier --mode train $( [ "$NO_AUGMENTATION" = true ] && echo "--no-augmentation" )
        ;;
    evaluate)
        if [ -z "$MODEL_DIR" ]; then
            # Recherche du dernier modèle entraîné
            MODEL_DIR=$(ls -d results/plantvillage_* 2>/dev/null | sort -r | head -n 1)
            if [ -z "$MODEL_DIR" ]; then
                echo "Erreur: Aucun modèle trouvé. Veuillez spécifier --model-dir ou entraîner un modèle d'abord."
                exit 1
            fi
            echo "Utilisation du modèle le plus récent: $MODEL_DIR"
        fi
        
        if [ ! -d "$MODEL_DIR" ]; then
            echo "Erreur: Le répertoire du modèle n'existe pas: $MODEL_DIR"
            exit 1
        fi
        
        echo "Évaluation du modèle dans $MODEL_DIR..."
        python -m scripts.plantvillage_classifier --mode evaluate --model-dir "$MODEL_DIR"
        ;;
    explain)
        if [ -z "$MODEL_DIR" ]; then
            # Recherche du dernier modèle entraîné
            MODEL_DIR=$(ls -d results/plantvillage_* 2>/dev/null | sort -r | head -n 1)
            if [ -z "$MODEL_DIR" ]; then
                echo "Erreur: Aucun modèle trouvé. Veuillez spécifier --model-dir ou entraîner un modèle d'abord."
                exit 1
            fi
            echo "Utilisation du modèle le plus récent: $MODEL_DIR"
        fi
        
        if [ ! -d "$MODEL_DIR" ]; then
            echo "Erreur: Le répertoire du modèle n'existe pas: $MODEL_DIR"
            exit 1
        fi
        
        echo "Génération d'explications avec le modèle dans $MODEL_DIR..."
        python -m scripts.plantvillage_classifier --mode explain --model-dir "$MODEL_DIR"
        ;;
    *)
        echo "Erreur: Commande inconnue: $COMMAND"
        show_help
        exit 1
        ;;
esac
