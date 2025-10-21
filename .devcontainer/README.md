# Dev Container GPU RTX 5070 - CUDA 12.8

Configuration complète pour utiliser TensorFlow/PyTorch sur GPU RTX 5070 avec CUDA 12.8.

## 📋 Prérequis

### Sur la machine hôte (Windows)

1. **Drivers NVIDIA** : Version compatible avec CUDA 12.8
   ```powershell
   nvidia-smi  # Vérifier la version du driver
   ```
   - Driver requis : >= 570.x pour CUDA 12.8

2. **Docker Desktop** avec support GPU
   - Installer Docker Desktop pour Windows
   - Activer l'intégration WSL2
   - Vérifier le support GPU :
     ```powershell
     docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
     ```

3. **VS Code** avec l'extension Dev Containers
   - Installer VS Code
   - Installer l'extension "Dev Containers" (ms-vscode-remote.remote-containers)

## 🚀 Utilisation

### Méthode 1 : Ouvrir dans VS Code

1. Ouvrir le projet dans VS Code
2. Appuyer sur `F1` ou `Ctrl+Shift+P`
3. Sélectionner : **"Dev Containers: Reopen in Container"**
4. Attendre la construction du container (5-10 minutes la première fois)
5. Le container s'ouvrira automatiquement avec l'environnement GPU activé

### Méthode 2 : Ligne de commande

```bash
# Depuis le répertoire du projet
cd c:/repository/datascience_rpojet_DS_2

# Construire le container
docker build -t gpu-dev-env .devcontainer/

# Lancer le container
docker run -it --gpus all --shm-size=4g \
  -v ${PWD}:/workspace \
  -v ~/.nv:/root/.nv \
  -p 8888:8888 -p 6006:6006 \
  gpu-dev-env
```

## 🧪 Vérification GPU

### Test automatique

Le script `test_gpu.py` s'exécute automatiquement au démarrage du container.
Pour le relancer manuellement :

```bash
python /workspace/.devcontainer/test_gpu.py
```

### Tests manuels

#### TensorFlow
```python
import tensorflow as tf

# Vérifier la version et les GPUs
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs disponibles: {len(tf.config.list_physical_devices('GPU'))}")
print(f"GPU détails: {tf.config.list_physical_devices('GPU')}")

# Test de calcul
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print(f"Calcul réussi: {c.shape}")
```

#### PyTorch
```python
import torch

# Vérifier CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test de calcul
device = torch.device('cuda')
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = torch.matmul(a, b)
print(f"Calcul réussi: {c.shape}, Device: {c.device}")
```

## 📦 Packages installés

### Deep Learning
- **TensorFlow** >= 2.20 (avec support CUDA 12.8)
- **PyTorch** >= 2.5 (avec torchvision, torchaudio)
- **Keras** >= 3.0
- **TensorRT** >= 8.6 (optimisation inférence)

### Data Science
- **NumPy**, **Pandas**, **Scikit-learn**
- **Matplotlib**, **Seaborn**, **Plotly**
- **OpenCV**, **Pillow**, **ImageIO**

### Outils ML
- **Jupyter Lab** 4.0+
- **TensorBoard**
- **Weights & Biases** (wandb)
- **MLflow**
- **HuggingFace** (transformers, datasets, accelerate)

## 🔧 Configuration

### Variables d'environnement

Les variables suivantes sont configurées automatiquement :

- `NVIDIA_VISIBLE_DEVICES=all` : Tous les GPUs visibles
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` : Capacités nécessaires
- `TF_FORCE_GPU_ALLOW_GROWTH=true` : TensorFlow alloue la mémoire à la demande
- `CUDA_CACHE_PATH=/root/.nv/ComputeCache` : Cache pour éviter la recompilation PTX

### Montages de volumes

- **Workspace** : `${workspaceFolder}` → `/workspace`
- **Cache GPU** : `~/.nv` → `/root/.nv` (évite la recompilation JIT PTX)
- **Cache packages** : `~/.cache` → `/root/.cache` (accélère les installations)

### Ports exposés

- **8888** : Jupyter Lab
- **6006** : TensorBoard

## 🎯 Lancer Jupyter Lab

```bash
# Dans le container
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Puis ouvrir le lien affiché dans le terminal (http://127.0.0.1:8888/lab?token=...)

## 🐛 Dépannage

### Problème : GPU non détecté

**Vérifications :**
1. Driver NVIDIA installé et à jour sur l'hôte
   ```bash
   nvidia-smi  # Sur l'hôte Windows
   ```

2. Docker a accès au GPU
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
   ```

3. Le container est lancé avec `--gpus=all`
   ```bash
   docker ps  # Vérifier les paramètres du container
   ```

### Problème : Erreur de mémoire GPU

**Solutions :**
- Augmenter `--shm-size` dans `devcontainer.json`
- Activer la croissance mémoire (déjà configuré avec `TF_FORCE_GPU_ALLOW_GROWTH`)
- Réduire la taille des batchs dans vos scripts

### Problème : Recompilation PTX (JIT) lente

**Solution :**
- Le cache GPU est monté dans `~/.nv/ComputeCache`
- Assurez-vous que ce dossier existe sur votre machine hôte
- La première exécution peut être lente, les suivantes utilisent le cache

### Problème : PEP 668 (pip install bloqué)

**Solution :**
- ✅ **Déjà résolu** : Tous les packages sont installés via Conda
- N'utilisez jamais `pip install` en système, ajoutez les packages dans `environment.yml`

## 📝 Ajouter des packages

### Via Conda (recommandé)

Modifier `.devcontainer/environment.yml` :

```yaml
dependencies:
  - votre-package>=version
```

Puis reconstruire le container.

### Via pip (si non disponible sur conda)

Ajouter dans la section `pip:` de `environment.yml` :

```yaml
dependencies:
  - pip:
    - votre-package>=version
```

## 🔄 Mettre à jour l'environnement

Si vous modifiez `environment.yml` :

1. Reconstruire le container dans VS Code :
   - `F1` → "Dev Containers: Rebuild Container"

2. Ou manuellement :
   ```bash
   docker build --no-cache -t gpu-dev-env .devcontainer/
   ```

## 📊 Monitoring GPU

### Avec nvidia-smi

```bash
# Monitoring en continu (toutes les 2 secondes)
watch -n 2 nvidia-smi

# Ou directement
nvidia-smi
```

### Avec PyTorch

```python
import torch
print(f"Mémoire allouée: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### Avec TensorFlow

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(tf.config.experimental.get_memory_info(gpu.name))
```

## ✅ Checklist de validation

- [ ] `nvidia-smi` affiche le GPU RTX 5070
- [ ] `tf.config.list_physical_devices('GPU')` retourne au moins 1 GPU
- [ ] `torch.cuda.is_available()` retourne `True`
- [ ] Test de calcul matriciel réussi sur GPU
- [ ] Jupyter Lab accessible sur http://localhost:8888
- [ ] Pas d'erreur PEP 668 (tout dans Conda)

## 📚 Ressources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)

---

**Configuration testée avec :**
- GPU : NVIDIA RTX 5070
- CUDA : 12.8.0 + cuDNN
- TensorFlow : >= 2.20
- PyTorch : >= 2.5
- Python : 3.12
