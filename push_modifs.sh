#!/bin/bash
#
# Script pour pousser les modifications sur GitHub (branche mpe_20251015)
#
echo "======================================"
echo "  Push modifications sur GitHub"
echo "======================================"
echo ""

# Afficher la branche actuelle
CURRENT_BRANCH=$(git branch --show-current)
echo "🌿 Branche actuelle: $CURRENT_BRANCH"
echo ""

# Vérifier le statut
echo "📋 Fichiers modifiés:"
git status --short
echo ""

# Afficher les principaux changements
echo "📝 Principaux fichiers modifiés:"
git status --porcelain | head -20
echo ""

# Demander confirmation
read -p "⚠️  Voulez-vous ajouter TOUS les fichiers modifiés ? (o/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Oo]$ ]]; then
    echo "❌ Opération annulée"
    echo ""
    echo "Pour ajouter seulement certains fichiers:"
    echo "  git add <fichier1> <fichier2> ..."
    echo "  git commit -m 'votre message'"
    echo "  git push origin $CURRENT_BRANCH"
    exit 1
fi

# Ajouter tous les fichiers
echo "➕ Ajout des fichiers..."
git add .
echo ""

# Demander le message de commit
echo "💬 Message de commit:"
echo "Suggestions:"
echo "  1) feat: nouvel entraînement multi-tâche RTX 5070"
echo "  2) feat: outputs v2 - Species 99.4%, Health 98.4%, Disease 94.3%"
echo "  3) update: résultats entraînement GPU 31 oct 2025"
echo ""
read -p "Entrez votre message (ou appuyez sur Entrée pour le message par défaut): " COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="feat: nouvel entraînement multi-tâche v2 RTX 5070

Résultats entraînement 31 octobre 2025:
- Species: 99.48% Accuracy, 99.41% F1
- Health: 98.74% Accuracy, 98.43% F1
- Disease: 95.61% Accuracy, 94.32% F1

Configuration:
- GPU: RTX 5070 Laptop (8GB)
- TensorFlow: 2.21.0-dev
- Epochs: 20 (Phase 1) + 10 (Phase 2 early stopping)
- Dataset: PlantVillage 54k images
"
fi

echo ""
echo "💾 Création du commit..."
git commit -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    echo "✓ Commit créé"
else
    echo "❌ Erreur lors du commit"
    exit 1
fi
echo ""

# Push
echo "======================================"
echo "  Prêt à pousser sur origin/$CURRENT_BRANCH"
echo "======================================"
echo ""
read -p "🚀 Pousser les modifications sur GitHub ? (o/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Oo]$ ]]; then
    echo "📤 Push en cours..."
    git push origin "$CURRENT_BRANCH"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Modifications poussées avec succès sur GitHub!"
        echo ""
        echo "📍 Branche: origin/$CURRENT_BRANCH"
        echo ""
    else
        echo "❌ Erreur lors du push"
        exit 1
    fi
else
    echo "ℹ️  Push annulé. Pour pousser manuellement:"
    echo "  git push origin $CURRENT_BRANCH"
fi

echo ""
echo "✅ Terminé!"
