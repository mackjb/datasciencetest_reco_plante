#!/bin/bash
#
# Commandes pour pousser la configuration GPU sur la branche mpe_20251015
#

echo "======================================"
echo "  Push Config GPU sur branche actuelle"
echo "======================================"
echo ""

# Afficher la branche actuelle
CURRENT_BRANCH=$(git branch --show-current)
echo "🌿 Branche actuelle: $CURRENT_BRANCH"
echo ""

# 1. Vérifier le statut
echo "📋 Étape 1/5 - Vérifier les fichiers modifiés:"
git status --short .devcontainer/
echo ""

# 2. Ajouter les fichiers
echo "➕ Étape 2/5 - Ajouter les fichiers au staging:"
git add .devcontainer/
echo "✓ Fichiers ajoutés"
echo ""

# 3. Afficher ce qui sera commité
echo "📝 Étape 3/5 - Fichiers qui seront commités:"
git diff --cached --name-only | grep devcontainer
echo ""

# 4. Demander confirmation avant commit
read -p "⚠️  Continuer avec le commit ? (o/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Oo]$ ]]; then
    echo "❌ Opération annulée"
    exit 1
fi

# 5. Commit
echo "💾 Étape 4/5 - Création du commit:"
git commit -m "feat: config GPU validée RTX 5070 (TF 2.21.0-dev, CUDA 12.9)

Configuration GPU testée et validée:
- GPU: NVIDIA GeForce RTX 5070 Laptop (8GB VRAM)
- Driver: 581.42, CUDA: 13.0
- TensorFlow: 2.21.0-dev20251013 (tf-nightly)
- CUDA packages: 12.9.x
- Test réussi: PlantVillage 1 epoch ~5min, 99.3% val accuracy

Fichiers ajoutés:
- Dockerfile.gpu: Image CUDA 12.8 + conda
- devcontainer-gpu.json: Config devcontainer avec GPU flags
- environment-gpu.yml: Définition conda env
- requirements-gpu-frozen.txt: Versions pip exactes
- test_gpu.py: Script de validation GPU
- README-GPU.md: Documentation complète
- QUICKSTART-GPU.md: Guide rapide
- save-gpu-config.sh: Script de sauvegarde/versioning
"

if [ $? -eq 0 ]; then
    echo "✓ Commit créé"
else
    echo "❌ Erreur lors du commit"
    exit 1
fi
echo ""

# 6. Créer le tag
echo "🏷️  Étape 5/5 - Création du tag:"
TAG_NAME="gpu-ok-rtx5070-$(date +%Y%m%d)"
git tag -a "$TAG_NAME" -m "GPU validé: RTX 5070, Driver 581.42, TF 2.21.0-dev

Configuration testée avec succès:
- Entraînement PlantVillage: 54k images, 13 classes
- Performance: ~5 min/epoch
- Val accuracy: 97.8% (stage 1), 99.3% (stage 2)
- VRAM: 1.7GB / 8GB
- GPU Util: 20-80%
"

if [ $? -eq 0 ]; then
    echo "✓ Tag créé: $TAG_NAME"
else
    echo "❌ Erreur lors de la création du tag"
    exit 1
fi
echo ""

# 7. Push (demander confirmation)
echo "======================================"
echo "  Prêt à pousser sur origin/$CURRENT_BRANCH"
echo "======================================"
echo ""
read -p "🚀 Pousser les modifications + tags sur origin/$CURRENT_BRANCH ? (o/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Oo]$ ]]; then
    echo "📤 Push en cours..."
    git push origin "$CURRENT_BRANCH" --tags
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Configuration GPU poussée avec succès!"
        echo ""
        echo "📍 Commit et tag disponibles sur: origin/$CURRENT_BRANCH"
        echo "🏷️  Tag: $TAG_NAME"
        echo ""
        echo "Pour récupérer cette config plus tard:"
        echo "  git checkout $TAG_NAME"
    else
        echo "❌ Erreur lors du push"
        exit 1
    fi
else
    echo "ℹ️  Push annulé. Pour pousser manuellement:"
    echo "  git push origin $CURRENT_BRANCH --tags"
fi
