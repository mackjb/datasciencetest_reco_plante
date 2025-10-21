# 🚀 Guide de Démarrage - Dev Container GPU RTX 5070

## 📝 Vue d'ensemble

Ce dev container vous permet d'utiliser votre **GPU NVIDIA RTX 5070** avec **CUDA 12.8** pour vos projets de Deep Learning avec TensorFlow et PyTorch.

---

## ⚡ Démarrage rapide (3 étapes)

### 1️⃣ Vérifier les prérequis

**Sur Windows, ouvrez PowerShell et exécutez :**

```powershell
# Vérifier le driver NVIDIA
nvidia-smi

# Vérifier Docker GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

✅ **Résultat attendu :** Vous devez voir les informations de votre GPU RTX 5070.

❌ **Si erreur :** Installez les drivers NVIDIA >= 570.x et Docker Desktop avec WSL2.

---

### 2️⃣ Ouvrir le projet dans VS Code

**Méthode A : Via VS Code (recommandé)**

1. Ouvrir VS Code dans le dossier du projet
2. Appuyer sur `F1`
3. Taper et sélectionner : **"Dev Containers: Reopen in Container"**
4. Attendre 5-10 minutes (première fois seulement)

**Méthode B : Via PowerShell**

```powershell
cd c:\repository\datascience_rpojet_DS_2\.devcontainer
.\dev-commands.ps1 build
.\dev-commands.ps1 run
```

---

### 3️⃣ Vérifier que le GPU fonctionne

**Une fois dans le container :**

```bash
# Test complet
python .devcontainer/test_gpu.py

# Ou test rapide
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

✅ **Résultat attendu :** Les GPUs sont détectés par TensorFlow et PyTorch.

---

## 📚 Utilisation quotidienne

### 🎓 Lancer Jupyter Lab

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Puis ouvrir dans votre navigateur : **http://localhost:8888**

---

### 📊 Lancer TensorBoard

```bash
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

Puis ouvrir : **http://localhost:6006**

---

### 🔍 Surveiller le GPU

```bash
# Monitoring en temps réel
watch -n 2 nvidia-smi

# Ou une seule fois
nvidia-smi
```

---

## 🧪 Exemples de code

### TensorFlow GPU

```python
import tensorflow as tf

# Vérifier les GPUs
print("GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))

# Calcul sur GPU
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print(f"Résultat: {c.shape}")
```

### PyTorch GPU

```python
import torch

# Vérifier CUDA
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Calcul sur GPU
device = torch.device('cuda')
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = torch.matmul(a, b)
print(f"Résultat: {c.shape}, Device: {c.device}")
```

---

## 📦 Gestion des packages

### ✅ FAIRE : Ajouter via Conda

1. Modifier `.devcontainer/environment.yml`
2. Ajouter votre package :
   ```yaml
   dependencies:
     - votre-package>=version
   ```
3. Reconstruire le container :
   - Dans VS Code : `F1` → "Dev Containers: Rebuild Container"
   - Ou : `.\dev-commands.ps1 rebuild`

### ❌ NE PAS FAIRE : pip install en système

```bash
pip install package  # ❌ VIOLE PEP 668
```

**Pourquoi ?** Ubuntu 24.04 protège l'environnement système Python. Utilisez **toujours Conda**.

---

## 🛠️ Commandes PowerShell

Script d'aide : `.devcontainer\dev-commands.ps1`

```powershell
# Construire l'image
.\dev-commands.ps1 build

# Lancer le container
.\dev-commands.ps1 run

# Ouvrir un shell
.\dev-commands.ps1 shell

# Tester le GPU
.\dev-commands.ps1 test-gpu

# Lancer Jupyter
.\dev-commands.ps1 jupyter

# Vérifier GPU hôte
.\dev-commands.ps1 check-gpu

# Nettoyer Docker
.\dev-commands.ps1 clean

# Rebuild complet
.\dev-commands.ps1 rebuild

# Aide
.\dev-commands.ps1 help
```

---

## 🐛 Problèmes courants

### ❓ GPU non détecté dans le container

**Diagnostic :**
```bash
# Dans le container
nvidia-smi
```

**Solutions :**
1. Vérifier driver NVIDIA sur Windows : `nvidia-smi`
2. Vérifier Docker GPU : `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
3. Reconstruire le container : `F1` → "Rebuild Container"
4. Vérifier `devcontainer.json` contient `"--gpus=all"` dans `runArgs`

---

### ❓ Erreur "Out of Memory" GPU

**Solutions :**
```python
# TensorFlow - Activer croissance mémoire (déjà configuré)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# PyTorch - Vider le cache
import torch
torch.cuda.empty_cache()

# Réduire la taille des batchs
batch_size = 16  # Au lieu de 32 ou 64
```

---

### ❓ Container lent au premier démarrage

**Normal !** Le premier build prend 5-10 minutes car il :
- Télécharge l'image CUDA (~2 GB)
- Installe Mambaforge
- Installe TensorFlow, PyTorch et tous les packages

Les démarrages suivants sont quasi instantanés.

---

### ❓ TensorFlow affiche des warnings CUDA

**Exemples de warnings normaux (pas d'erreurs) :**
```
PTX JIT compilation...
Could not find cuda drivers...
```

**Ces warnings sont OK** tant que `tf.config.list_physical_devices('GPU')` détecte le GPU.

Le cache PTX est monté dans `~/.nv` pour accélérer les compilations suivantes.

---

## 📁 Structure des fichiers

```
.devcontainer/
├── Dockerfile              # Image Docker CUDA 12.8
├── devcontainer.json       # Config VS Code + GPU
├── environment.yml         # Packages Conda/pip
├── test_gpu.py            # Script de test GPU
├── example_gpu_test.ipynb # Notebook de démo
├── dev-commands.ps1       # Script PowerShell
├── quick_start.sh         # Script bash de démarrage
├── README.md              # Documentation complète
├── QUICK_REFERENCE.md     # Référence rapide
├── GETTING_STARTED.md     # Ce fichier
└── .dockerignore          # Fichiers à ignorer
```

---

## 🎯 Checklist de validation

Avant de commencer vos projets, vérifiez :

- [ ] `nvidia-smi` affiche le RTX 5070
- [ ] `tf.config.list_physical_devices('GPU')` retourne 1+ GPU
- [ ] `torch.cuda.is_available()` retourne `True`
- [ ] Test matriciel GPU réussi (plus rapide que CPU)
- [ ] Jupyter Lab accessible sur http://localhost:8888
- [ ] Environnement Conda `gpu-env` activé
- [ ] Pas d'erreur PEP 668

---

## 🎓 Ressources et documentation

### Documentation locale
- **Guide complet** : `.devcontainer/README.md`
- **Référence rapide** : `.devcontainer/QUICK_REFERENCE.md`
- **Notebook exemple** : `.devcontainer/example_gpu_test.ipynb`

### Documentation externe
- [NVIDIA CUDA Docs](https://docs.nvidia.com/cuda/)
- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)

---

## 💡 Conseils et bonnes pratiques

### Performance GPU

1. **Batch size** : Augmentez pour mieux utiliser le GPU (32, 64, 128)
2. **Mixed Precision** : Utilisez FP16 pour plus de vitesse
   ```python
   # TensorFlow
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   
   # PyTorch
   from torch.cuda.amp import autocast
   with autocast():
       output = model(input)
   ```
3. **Préchargement données** : Utilisez `prefetch` pour éviter d'attendre le GPU
   ```python
   # TensorFlow
   dataset = dataset.prefetch(tf.data.AUTOTUNE)
   ```

### Organisation du code

1. **Versionnez vos expériences** : Utilisez MLflow ou W&B
2. **Sauvegardez régulièrement** : Checkpoints tous les N epochs
3. **Utilisez TensorBoard** : Monitoring en temps réel

### Sécurité

1. **API Keys** : Utilisez `.env` (ajoutez dans `.gitignore`)
2. **Données sensibles** : Ne committez jamais les datasets
3. **Modèles entraînés** : Stockez en externe (pas dans Git)

---

## 🆘 Support

### En cas de problème

1. **Consultez README.md** : Documentation détaillée
2. **Vérifiez QUICK_REFERENCE.md** : Solutions rapides
3. **Testez** : `python .devcontainer/test_gpu.py`
4. **Logs** : `.\dev-commands.ps1 logs`

### Informations utiles pour le debug

```bash
# Version Python
python --version

# Packages installés
conda list

# Info GPU
nvidia-smi

# Info Docker
docker --version
docker info | grep -i runtime
```

---

## ✅ Prêt à commencer !

Votre environnement GPU est maintenant configuré et prêt pour :

- 🧠 **Deep Learning** : TensorFlow, PyTorch, Keras
- 🖼️ **Computer Vision** : OpenCV, PIL, ImageIO
- 📊 **Data Science** : NumPy, Pandas, Scikit-learn
- 📈 **Visualisation** : Matplotlib, Seaborn, Plotly
- 🤗 **NLP** : HuggingFace Transformers
- 🚀 **MLOps** : TensorBoard, MLflow, W&B

**Bon coding ! 🎉**

---

**Configuration** : NVIDIA RTX 5070 | CUDA 12.8.0 | TensorFlow ≥2.20 | PyTorch ≥2.5 | Python 3.12
