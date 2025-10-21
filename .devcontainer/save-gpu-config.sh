#!/bin/bash
#
# Script pour sauvegarder et versionner la configuration GPU validée
#
# Usage: bash .devcontainer/save-gpu-config.sh
#

set -e

echo "======================================"
echo "  Sauvegarde Config GPU - RTX 5070"
echo "======================================"
echo ""

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier qu'on est dans le bon répertoire
if [ ! -d ".devcontainer" ]; then
    echo -e "${RED}❌ Erreur: Lancez ce script depuis la racine du projet${NC}"
    exit 1
fi

# Afficher les versions installées
echo -e "${YELLOW}📦 Versions actuelles:${NC}"
echo ""
conda run -n gpu-env python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'GPU détecté: {len(tf.config.list_physical_devices(\"GPU\"))} device(s)')
"
echo ""

# Générer requirements figé
echo -e "${YELLOW}💾 Génération requirements-gpu-frozen.txt...${NC}"
conda run -n gpu-env pip freeze > .devcontainer/requirements-gpu-frozen.txt
echo -e "${GREEN}✓${NC} Fichier créé"
echo ""

# Exporter l'environnement conda complet
echo -e "${YELLOW}💾 Export de l'environnement conda...${NC}"
conda env export -n gpu-env > .devcontainer/environment-gpu-export.yml
echo -e "${GREEN}✓${NC} environment-gpu-export.yml créé"
echo ""

# Créer un rapport de validation
REPORT_FILE=".devcontainer/validation-report-$(date +%Y%m%d-%H%M%S).txt"
echo -e "${YELLOW}📋 Création rapport de validation...${NC}"

cat > "$REPORT_FILE" << EOF
===========================================
  Rapport de Validation GPU
  Date: $(date -Iseconds)
===========================================

SYSTÈME
-------
OS: $(lsb_release -d | cut -f2)
Kernel: $(uname -r)

GPU HÔTE
--------
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)

NVIDIA-SMI
----------
$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv)

ENVIRONNEMENT CONDA
-------------------
Nom: gpu-env
Localisation: $(conda env list | grep gpu-env | awk '{print $2}')

TENSORFLOW
----------
Version: $(conda run -n gpu-env python -c "import tensorflow as tf; print(tf.__version__)")
GPU support: $(conda run -n gpu-env python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())")
GPU détectés: $(conda run -n gpu-env python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")

CUDA PACKAGES (pip)
-------------------
$(conda run -n gpu-env pip list | grep -E "tensorflow|nvidia-cuda|nvidia-cublas|nvidia-cudnn|keras")

PACKAGES PRINCIPAUX
-------------------
$(conda run -n gpu-env pip list | grep -E "scikit-learn|pandas|matplotlib|pillow")

TEST GPU
--------
$(conda run -n gpu-env python .devcontainer/test_gpu.py 2>&1 | tail -20)

FICHIERS SAUVEGARDÉS
--------------------
- environment-gpu.yml (définition manuelle)
- environment-gpu-export.yml (export conda complet)
- requirements-gpu-frozen.txt (pip freeze)
- Dockerfile.gpu
- devcontainer-gpu.json
- test_gpu.py
- README-GPU.md
- $REPORT_FILE
EOF

echo -e "${GREEN}✓${NC} Rapport: $REPORT_FILE"
echo ""

# Afficher les fichiers à commiter
echo -e "${YELLOW}📁 Fichiers à versionner:${NC}"
ls -lh .devcontainer/*.{yml,txt,json,py,sh,md} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Proposer le commit Git
echo -e "${YELLOW}🔖 Commandes Git suggérées:${NC}"
echo ""
echo "  git add .devcontainer/"
echo "  git commit -m \"feat: config GPU validée RTX 5070 (TF $(conda run -n gpu-env python -c 'import tensorflow as tf; print(tf.__version__)'), CUDA 12.9)\""
echo "  git tag -a gpu-ok-rtx5070-$(date +%Y%m%d) -m \"GPU validé: RTX 5070, Driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)\""
echo "  git push origin main --tags"
echo ""

echo -e "${GREEN}✅ Configuration GPU sauvegardée avec succès!${NC}"
echo ""
echo "Pour recréer l'environnement plus tard:"
echo "  1. conda env create -f .devcontainer/environment-gpu.yml"
echo "  2. Ou: pip install -r .devcontainer/requirements-gpu-frozen.txt"
echo "  3. Tester: python .devcontainer/test_gpu.py"
