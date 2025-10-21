#!/bin/bash
# Script de démarrage rapide pour le devcontainer GPU

echo "🚀 Démarrage rapide - Dev Container GPU RTX 5070"
echo "=================================================="
echo ""

# Activation de l'environnement conda
source /opt/conda/etc/profile.d/conda.sh
conda activate gpu-env

# Affichage des informations système
echo "📊 Informations système:"
echo "  Python: $(python --version)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Vérification GPU
echo "🎮 GPU disponible:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Test rapide GPU
echo "🧪 Test rapide des frameworks..."
python -c "
import tensorflow as tf
import torch
print(f'✅ TensorFlow {tf.__version__} - GPU: {len(tf.config.list_physical_devices(\"GPU\"))} détecté(s)')
print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')
"
echo ""

# Options disponibles
echo "📋 Commandes disponibles:"
echo "  1. Lancer le test complet GPU:"
echo "     python .devcontainer/test_gpu.py"
echo ""
echo "  2. Démarrer Jupyter Lab:"
echo "     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "  3. Démarrer TensorBoard:"
echo "     tensorboard --logdir=./logs --host=0.0.0.0 --port=6006"
echo ""
echo "  4. Monitoring GPU en continu:"
echo "     watch -n 2 nvidia-smi"
echo ""
echo "=================================================="
echo "✅ Environnement GPU prêt!"
