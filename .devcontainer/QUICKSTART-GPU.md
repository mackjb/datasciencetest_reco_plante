# 🚀 Guide Rapide - Configuration GPU

## 📋 Fichiers de configuration créés

```
.devcontainer/
├── Dockerfile.gpu                    # Image Docker CUDA 12.8 + conda
├── devcontainer-gpu.json             # Config VSCode/Windsurf devcontainer
├── environment-gpu.yml               # Définition environnement conda (manuel)
├── requirements-gpu-frozen.txt       # Versions exactes pip (auto-généré)
├── test_gpu.py                       # Script de test GPU
├── save-gpu-config.sh                # Script de sauvegarde/versioning
├── README-GPU.md                     # Documentation complète
└── QUICKSTART-GPU.md                 # Ce fichier
```

---

## ⚡ Utilisation immédiate

### Option 1: Devcontainer (Recommandé pour VSCode/Windsurf)

1. **Ouvrir le projet dans VSCode/Windsurf**

2. **Lancer le devcontainer**:
   - `Ctrl+Shift+P` (ou `Cmd+Shift+P` sur Mac)
   - Chercher: `Dev Containers: Rebuild and Reopen in Container`
   - Sélectionner le fichier: `devcontainer-gpu.json`

3. **Tester le GPU**:
   ```bash
   python .devcontainer/test_gpu.py
   ```

4. **Lancer un entraînement**:
   ```bash
   conda activate gpu-env
   python train_species_plantvillage_keras.py --epochs 1 --batch_size 64
   ```

### Option 2: Conda local (Sans Docker)

```bash
# 1. Créer l'environnement
conda env create -f .devcontainer/environment-gpu.yml

# 2. Activer
conda activate gpu-env

# 3. Tester
python .devcontainer/test_gpu.py

# 4. Entraîner
python train_species_plantvillage_keras.py --epochs 1 --batch_size 64
```

### Option 3: Pip pur (Versions exactes figées)

```bash
# 1. Créer un venv
python3.11 -m venv venv-gpu
source venv-gpu/bin/activate

# 2. Installer les versions exactes
pip install -r .devcontainer/requirements-gpu-frozen.txt

# 3. Tester
python .devcontainer/test_gpu.py
```

---

## 💾 Sauvegarder et versionner votre configuration

### Après validation de votre GPU:

```bash
# 1. Exécuter le script de sauvegarde
bash .devcontainer/save-gpu-config.sh

# 2. Suivre les instructions Git affichées
# Exemple de sortie:
#   git add .devcontainer/
#   git commit -m "feat: config GPU validée RTX 5070"
#   git tag -a gpu-ok-rtx5070-20251017 -m "GPU validé"
#   git push origin mpe_20251015 --tags  # Sur VOTRE branche
```

Le script génère automatiquement:
- ✅ `requirements-gpu-frozen.txt` (versions pip exactes)
- ✅ `environment-gpu-export.yml` (export conda complet)
- ✅ `validation-report-YYYYMMDD-HHMMSS.txt` (rapport de test)

---

## 🔄 Reproduire la configuration plus tard

### Sur une nouvelle machine / nouveau container:

#### Méthode A: Conda (Recommandé)
```bash
# Cloner le repo avec le tag spécifique
git clone https://github.com/votre-repo/datasciencetest_reco_plante.git
cd datasciencetest_reco_plante
git checkout gpu-ok-rtx5070-20251017  # Utiliser votre tag

# Créer l'environnement
conda env create -f .devcontainer/environment-gpu.yml

# Tester
conda activate gpu-env
python .devcontainer/test_gpu.py
```

#### Méthode B: Docker devcontainer
```bash
# 1. Cloner + checkout tag
git clone ... && cd ... && git checkout gpu-ok-rtx5070-20251017

# 2. Ouvrir dans VSCode
code .

# 3. Reopen in Container
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

#### Méthode C: Pip pur
```bash
# Versions exactes figées
pip install -r .devcontainer/requirements-gpu-frozen.txt
```

---

## ✅ Checklist de validation GPU

Avant de commiter votre configuration, vérifiez:

- [ ] `nvidia-smi` fonctionne et affiche votre GPU
- [ ] `python test_gpu.py` passe tous les tests
- [ ] Un entraînement court (1 epoch) se termine sans erreur
- [ ] La GPU est bien utilisée (vérifier avec `nvidia-smi` pendant l'entraînement)
- [ ] `requirements-gpu-frozen.txt` est à jour
- [ ] Tag Git créé avec description claire

---

## 🐛 Troubleshooting rapide

### Problème: "No GPU detected"
```bash
# Vérifier driver
nvidia-smi

# Vérifier Docker GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Vérifier variables d'env
echo $CUDA_VISIBLE_DEVICES  # Doit être vide ou "all"
```

### Problème: "CUDA_ERROR_INVALID_PTX"
```bash
# Installer le compilateur nvcc
conda activate gpu-env
pip install nvidia-cuda-nvcc-cu12==12.9.86

# Définir les variables d'environnement
export CUDA_FORCE_PTX_JIT=1
export CUDA_CACHE_MAXSIZE=2147483648
```

### Problème: Compilation PTX très lente (>10 min)
✅ **Normal pour RTX 5070 (première compilation)**

Patience, les kernels sont compilés à la volée et mis en cache.
Les runs suivants seront rapides (~30s startup).

### Problème: Out of Memory
```bash
# Réduire batch size
python train_species_plantvillage_keras.py --batch_size 32

# Activer croissance mémoire progressive
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

---

## 📚 Documentation complète

Pour plus de détails, voir:
- **[README-GPU.md](.devcontainer/README-GPU.md)** - Documentation technique complète
- **[test_gpu.py](.devcontainer/test_gpu.py)** - Script de validation
- **[Dockerfile.gpu](.devcontainer/Dockerfile.gpu)** - Configuration Docker

---

## 🎯 Commandes utiles

```bash
# Tester le GPU
python .devcontainer/test_gpu.py

# Monitorer le GPU pendant l'entraînement
watch -n 1 nvidia-smi

# Voir les versions installées
conda run -n gpu-env pip list | grep -E "tensorflow|nvidia"

# Sauvegarder la config actuelle
bash .devcontainer/save-gpu-config.sh

# Comparer les environnements conda
conda env export -n gpu-env > current-env.yml
diff .devcontainer/environment-gpu.yml current-env.yml
```

---

## 📊 Performances attendues (RTX 5070)

| Métrique | Valeur |
|----------|--------|
| Startup TensorFlow | ~30s (après première compilation) |
| Compilation kernels (première fois) | 5-15 min |
| Entraînement PlantVillage (1 epoch) | ~5 min |
| VRAM utilisée | 1.7-2.5 GB / 8 GB |
| GPU Utilization | 20-80% |
| Validation accuracy (1 epoch) | >97% |

---

## 📞 Support

En cas de problème:
1. Vérifier [README-GPU.md](.devcontainer/README-GPU.md) section Troubleshooting
2. Exécuter `bash .devcontainer/save-gpu-config.sh` pour générer un rapport
3. Consulter le rapport de validation généré

---

**Dernière mise à jour**: 17 octobre 2025  
**GPU validée**: NVIDIA GeForce RTX 5070 Laptop (8GB)  
**TensorFlow**: 2.21.0-dev20251013  
**CUDA**: 12.9
