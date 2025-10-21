# 🚀 Configuration GPU validée - RTX 5070

## ✅ Statut de validation

- **Date de validation**: 16-17 octobre 2025
- **GPU testée**: NVIDIA GeForce RTX 5070 Laptop (8GB VRAM)
- **Compute Capability**: 12.0a
- **Driver hôte**: 581.42
- **CUDA hôte**: 13.0
- **Test d'entraînement**: ✅ PlantVillage (54k images, 13 classes, 1 epoch ~5 min)

---

## 📦 Versions exactes des packages (testées et validées)

```yaml
# TensorFlow GPU (nightly requis pour RTX 5070 - SM 12.0)
tf_nightly==2.21.0.dev20251013
tb-nightly==2.20.0a20250717
keras==3.11.3

# NVIDIA CUDA packages (compilation JIT-PTX pour SM 12.0)
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cublas-cu12==12.9.1.4
nvidia-cudnn-cu12==9.14.0.64
nvidia-cuda-nvcc-cu12==12.9.86
```

### ⚠️ Pourquoi tf-nightly et non tensorflow stable ?

**RTX 5070 a une compute capability 12.0a** (architecture Blackwell), introduite en 2024.

- **tensorflow stable (2.18-2.20)**: N'inclut pas de binaires CUDA précompilés pour SM 12.0
  - Résultat: Compilation JIT-PTX très lente (10-30 min au premier lancement)
  - Erreurs possibles: `CUDA_ERROR_INVALID_PTX`, `CUDA_ERROR_INVALID_HANDLE`
  
- **tf-nightly**: Support expérimental pour les nouvelles architectures GPU
  - Compile mieux les kernels PTX pour SM 12.0
  - Moins d'erreurs lors de la compilation JIT

---

## 🐳 Reproduction de l'environnement

### Méthode 1: Avec devcontainer (recommandé)

1. **Prérequis hôte**:
   ```bash
   # Vérifier le driver NVIDIA
   nvidia-smi
   # Doit afficher: Driver >= 580.x, CUDA >= 12.8
   
   # Installer nvidia-container-toolkit si nécessaire
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Lancer le devcontainer**:
   ```bash
   # Dans VS Code / Windsurf:
   # 1. Ouvrir le workspace
   # 2. Cmd/Ctrl + Shift + P
   # 3. "Dev Containers: Reopen in Container"
   # 4. Sélectionner devcontainer-gpu.json
   ```

3. **Tester le GPU**:
   ```bash
   python test_gpu.py
   ```

### Méthode 2: Conda manuel

```bash
# Créer l'environnement
conda env create -f environment-gpu.yml

# Activer
conda activate gpu-env

# Tester
python test_gpu.py
```

---

## 🔧 Variables d'environnement GPU (critiques)

Ajoutez ces variables à votre shell ou `devcontainer.json`:

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true        # Allocation mémoire progressive
export CUDA_CACHE_MAXSIZE=2147483648         # Cache 2GB pour kernels compilés
export CUDA_FORCE_PTX_JIT=1                  # Force compilation PTX (SM 12.0)
export CUDA_MODULE_LOADING=LAZY              # Lazy loading des modules CUDA
export TF_CPP_MIN_LOG_LEVEL=2                # Réduire les warnings TF
```

### Pourquoi ces variables ?

- **TF_FORCE_GPU_ALLOW_GROWTH**: Évite l'allocation de toute la VRAM au démarrage (important pour GPU 8GB)
- **CUDA_CACHE_MAXSIZE**: Stocke les kernels compilés (évite la recompilation à chaque run)
- **CUDA_FORCE_PTX_JIT**: Force la compilation Just-In-Time du PTX → binaires SM 12.0
- **CUDA_MODULE_LOADING=LAZY**: Charge les modules CUDA à la demande (startup plus rapide)

---

## 🧪 Tests de validation

### Test 1: Détection GPU
```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
```

**Sortie attendue**:
```
TensorFlow: 2.21.0-dev20251013
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Test 2: Calcul sur GPU
```python
import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)
print(f"✅ Résultat: {c.shape}")
```

### Test 3: Entraînement réel (PlantVillage)
```bash
# Utiliser l'environnement gpu-env
conda activate gpu-env

# Lancer un entraînement de test (1 epoch)
python train_species_plantvillage_keras.py \
  --epochs 1 \
  --batch_size 64 \
  --output_dir outputs_gpu_test \
  --keep_list_csv /dev/null

# Durée attendue: ~5 min/epoch (RTX 5070)
# Val accuracy attendue: >97% après 1 epoch
```

---

## 🏷️ Versioning Git

```bash
# Sauvegarder la configuration GPU validée
git add .devcontainer/
git commit -m "feat: devcontainer GPU validé (RTX 5070, TF 2.21, CUDA 12.9)"
git tag -a gpu-ok-rtx5070 -m "Configuration GPU validée sur RTX 5070 Laptop"
git push origin main --tags
```

### Retrouver cette configuration plus tard:
```bash
git checkout gpu-ok-rtx5070
# Ou créer une branche depuis ce tag
git checkout -b gpu-stable gpu-ok-rtx5070
```

---

## 📊 Performances mesurées (RTX 5070)

| Métrique | Valeur |
|----------|--------|
| **Dataset** | PlantVillage color (54,305 images) |
| **Modèle** | DenseNet121 (transfer learning) |
| **Batch size** | 64 |
| **Durée/epoch** | ~5 minutes |
| **VRAM utilisée** | ~1.7 GB / 8 GB |
| **GPU Util** | 20-80% (varie selon la phase) |
| **Val accuracy (epoch 1)** | 97.8% (stage 1), 99.3% (stage 2) |

---

## 🐛 Troubleshooting

### Erreur: "No GPU detected"
```bash
# Vérifier le driver NVIDIA
nvidia-smi

# Vérifier que Docker voit le GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Vérifier nvidia-container-toolkit
sudo systemctl status nvidia-container-toolkit
```

### Erreur: "CUDA_ERROR_INVALID_PTX"
```bash
# Installer nvidia-cuda-nvcc-cu12 (compilateur PTX)
conda activate gpu-env
pip install nvidia-cuda-nvcc-cu12==12.9.86

# Vérifier les variables d'environnement
export CUDA_FORCE_PTX_JIT=1
export CUDA_CACHE_MAXSIZE=2147483648
```

### Compilation PTX très lente (10-30 min au premier lancement)
✅ **C'est normal pour RTX 5070 (SM 12.0)**
- TensorFlow compile les kernels CUDA à la volée (JIT)
- Les kernels compilés sont mis en cache
- Les lancements suivants seront rapides (~30s startup)

**Accélérer**:
```bash
# Augmenter le cache
export CUDA_CACHE_MAXSIZE=4294967296  # 4GB

# Pré-compiler en lançant test_gpu.py
python test_gpu.py
```

### Out of Memory (OOM)
```bash
# Réduire le batch size
python train_species_plantvillage_keras.py --batch_size 32

# Ou activer la croissance mémoire progressive
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

---

## 📚 Références

- [NVIDIA RTX 5070 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
- [Dev Containers GPU](https://code.visualstudio.com/remote/advancedcontainers/gpu-support)

---

## 📝 Notes de compatibilité

### Image Docker base
- ✅ `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`
- ✅ `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04` (alternative)

### Alternatives à tf-nightly

Si vous préférez une version stable (avec compromis sur performance):

```bash
# TensorFlow stable + packages CUDA
pip install tensorflow==2.20.0
pip install nvidia-cuda-runtime-cu12 \
            nvidia-cublas-cu12 \
            nvidia-cudnn-cu12 \
            nvidia-cuda-nvcc-cu12

# ⚠️ Attendez-vous à:
# - Compilation PTX plus lente (15-30 min au premier run)
# - Possibles erreurs PTX sur certaines opérations
```

**Recommandation**: Rester sur `tf-nightly` jusqu'à ce que TensorFlow 2.21 stable soit publié avec support SM 12.0.
