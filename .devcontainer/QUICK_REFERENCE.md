# 📋 Référence Rapide - Dev Container GPU

## 🚀 Démarrage rapide

### Ouvrir dans VS Code
```
F1 → "Dev Containers: Reopen in Container"
```

### Vérifier le GPU
```bash
# Voir le GPU
nvidia-smi

# Test complet
python .devcontainer/test_gpu.py

# Test TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Test PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📦 Environnement Conda

### Activer l'environnement
```bash
conda activate gpu-env
```

### Ajouter un package
```bash
# Modifier environment.yml, puis:
conda env update -f .devcontainer/environment.yml
```

### Lister les packages
```bash
conda list
pip list
```

## 🎓 Jupyter Lab

### Démarrer
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Accéder
```
http://localhost:8888
```

### Arrêter
```bash
# Ctrl+C dans le terminal
```

## 📊 TensorBoard

### Démarrer
```bash
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

### Accéder
```
http://localhost:6006
```

## 🔍 Monitoring GPU

### Temps réel
```bash
watch -n 2 nvidia-smi
```

### Avec PyTorch
```python
import torch
print(f"Allouée: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
print(f"Totale: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
```

### Avec TensorFlow
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.get_memory_info('GPU:0')
```

## 🧪 Tests rapides

### TensorFlow sur GPU
```python
import tensorflow as tf

with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print(f"✅ Calcul réussi: {c.shape}")
```

### PyTorch sur GPU
```python
import torch

device = torch.device('cuda')
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = torch.matmul(a, b)
print(f"✅ Calcul réussi: {c.shape}, Device: {c.device}")
```

## 🐛 Dépannage express

### GPU non détecté
```bash
# 1. Vérifier driver sur hôte
nvidia-smi  # Sur Windows

# 2. Vérifier Docker GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# 3. Rebuild container
# F1 → "Dev Containers: Rebuild Container"
```

### Erreur mémoire GPU
```python
# TensorFlow: activer croissance mémoire (déjà fait)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# PyTorch: vider cache
import torch
torch.cuda.empty_cache()
```

### Package manquant
```bash
# NE PAS FAIRE: pip install package (viole PEP 668)
# FAIRE: Ajouter dans environment.yml + rebuild
```

## 📁 Fichiers importants

| Fichier | Description |
|---------|-------------|
| `Dockerfile` | Image Docker CUDA 12.8 + Conda |
| `devcontainer.json` | Configuration VS Code + GPU |
| `environment.yml` | Packages Python/Conda |
| `test_gpu.py` | Script de test complet |
| `example_gpu_test.ipynb` | Notebook de démonstration |
| `README.md` | Documentation complète |

## 🔗 Commandes Docker manuelles

### Build
```bash
cd c:/repository/datascience_rpojet_DS_2
docker build -t gpu-dev-env .devcontainer/
```

### Run
```bash
docker run -it --gpus all --shm-size=4g \
  -v ${PWD}:/workspace \
  -v ~/.nv:/root/.nv \
  -p 8888:8888 -p 6006:6006 \
  gpu-dev-env
```

### Clean
```bash
# Supprimer containers arrêtés
docker container prune

# Supprimer images non utilisées
docker image prune

# Tout nettoyer (ATTENTION!)
docker system prune -a
```

## ⚡ Variables d'environnement clés

| Variable | Valeur | Description |
|----------|--------|-------------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | Tous les GPUs visibles |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Capacités GPU |
| `TF_FORCE_GPU_ALLOW_GROWTH` | `true` | Mémoire dynamique TF |
| `CUDA_CACHE_PATH` | `/root/.nv/ComputeCache` | Cache PTX |

## 📚 Ressources

- **Documentation complète**: `.devcontainer/README.md`
- **Test GPU**: `python .devcontainer/test_gpu.py`
- **Notebook exemple**: `.devcontainer/example_gpu_test.ipynb`
- **CUDA Docs**: https://docs.nvidia.com/cuda/
- **TensorFlow GPU**: https://www.tensorflow.org/install/gpu
- **PyTorch CUDA**: https://pytorch.org/docs/stable/notes/cuda.html

---

**Version**: CUDA 12.8.0 | TensorFlow ≥2.20 | PyTorch ≥2.5 | Python 3.12
