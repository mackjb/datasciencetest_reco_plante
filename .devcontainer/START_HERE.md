# 🚀 Démarrage Dev Container GPU

## ✅ Fichiers corrigés

Les 4 fichiers essentiels ont été créés/corrigés :

1. **`Dockerfile`** 
   - ✅ Base: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`
   - ✅ Packages Ubuntu 24.04 compatibles (`libgl1` au lieu de `libgl1-mesa-glx`)
   - ✅ Miniforge + environnement Conda `gpu-env`

2. **`devcontainer.json`**
   - ✅ GPU: `--gpus=all`, `--shm-size=4g`, `--ipc=host`
   - ✅ Montage Windows corrigé: `${localEnv:USERPROFILE}${localEnv:HOME}/.nv`
   - ✅ Activation automatique de l'environnement Conda

3. **`environment.yml`**
   - ✅ Python 3.12
   - ✅ TensorFlow ≥2.17, PyTorch ≥2.4
   - ✅ **SANS** cuda-toolkit/cudnn (évite les doublons avec l'image Docker)
   - ✅ Packages data science essentiels

4. **`test_gpu.py`**
   - ✅ Affiche `tf.__version__`
   - ✅ Liste les GPU détectés
   - ✅ Teste `matmul` sur `/GPU:0`

---

## 🎯 Démarrage (Windsurf/VS Code)

### Méthode simple (recommandée)

1. **Fermer complètement Windsurf**

2. **Rouvrir le projet**
   ```
   Ouvrir: c:\repository\datascience_rpojet_DS_2
   ```

3. **Ouvrir dans le container**
   - Appuyer sur `F1`
   - Sélectionner: **"Dev Containers: Reopen in Container"**
   
4. **Premier build** (5-10 minutes)
   - Le container va se construire automatiquement
   - Patience pendant le téléchargement et l'installation

5. **Vérification automatique**
   - Le script `test_gpu.py` s'exécute automatiquement
   - Vous devez voir: `✅ GPU fonctionnel!`

---

## 🧪 Test manuel

Si vous voulez retester le GPU :

```bash
python /workspace/.devcontainer/test_gpu.py
```

**Résultat attendu :**
```
============================================================
  TEST GPU - RTX 5070 CUDA 12.8
============================================================

TensorFlow version: 2.17.x (ou supérieur)
Built with CUDA: True

GPUs détectés: 1
  GPU 0: /physical_device:GPU:0
    Compute Capability: (12, 0)

Test matmul sur /GPU:0...
✅ Calcul réussi! Shape: (1000, 1000)

✅ GPU fonctionnel!
```

---

## 📓 Lancer Jupyter Lab

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Puis ouvrir dans votre navigateur : **http://localhost:8888**

---

## ❌ Si erreur de build

### Problème : Erreur APT "libgl1-mesa-glx"
**Solution** : ✅ Déjà corrigé ! Utilisé `libgl1` compatible Ubuntu 24.04

### Problème : Montage `.nv` échoue
**Solution** : ✅ Déjà corrigé ! Montage Windows sûr dans `devcontainer.json`

### Problème : CUDA/cuDNN en conflit
**Solution** : ✅ Déjà corrigé ! Pas de cuda-toolkit dans `environment.yml`

### Rebuild complet

Si nécessaire, forcer un rebuild :

**Dans Windsurf/VS Code:**
```
F1 → "Dev Containers: Rebuild Container"
```

**Ou en PowerShell:**
```powershell
cd c:\repository\datascience_rpojet_DS_2\.devcontainer
docker build --no-cache -t vsc-datascience-gpu .
```

---

## 🔍 Vérification GPU Windows

Avant de démarrer, vérifiez que votre GPU est accessible :

```powershell
# Vérifier driver NVIDIA
nvidia-smi

# Vérifier Docker GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

✅ Vous devez voir votre **RTX 5070**

---

## 📦 Environnement Conda

L'environnement `gpu-env` est activé automatiquement au démarrage.

### Vérifier l'environnement actif
```bash
conda info --envs
# Doit montrer: * gpu-env
```

### Ajouter un package
1. Modifier `.devcontainer/environment.yml`
2. Rebuild le container (`F1` → Rebuild Container)

### ❌ NE PAS FAIRE
```bash
pip install package  # ❌ Viole PEP 668
```

### ✅ FAIRE
Ajouter dans `environment.yml` puis rebuild

---

## 🎓 Prochaines étapes

1. ✅ Container démarré et GPU détecté
2. ✅ Test GPU réussi
3. 🚀 Lancer Jupyter Lab
4. 🚀 Commencer vos projets Deep Learning !

---

## 📚 Ressources

- **Documentation complète** : `.devcontainer/README.md`
- **Référence rapide** : `.devcontainer/QUICK_REFERENCE.md`
- **Notebook exemple** : `.devcontainer/example_gpu_test.ipynb`

---

**Configuration** : RTX 5070 | CUDA 12.8 | TensorFlow ≥2.17 | Python 3.12
