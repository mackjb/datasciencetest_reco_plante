Reconnaissance de plantes et de maladies sur PlantVillage
========================================================

Ce dépôt correspond à un **projet d’étude en data science / machine learning / deep learning** autour du
dataset d’images **PlantVillage**. L’objectif est de comparer plusieurs approches pour :

- **Identifier l’espèce** de la plante à partir d’une image de feuille.
- **Détecter si la plante est saine ou malade**.
- **Identifier la maladie** lorsqu’elle est présente.

Le projet combine :

- **Modèles de Machine Learning classiques** entraînés sur des **descripteurs d’images** (features tabulaires).
- **Architectures Deep Learning** (EfficientNetV2S, mono‑tâche, multi‑tâches, tête unique 35 classes).
- **Modèles YOLOv8 en classification**.
- **Interprétabilité** via **SHAP** (ML) et **Grad‑CAM** (DL).


## 1. Données

### 1.1 Dataset PlantVillage

Le projet s’appuie sur les variantes (classiques) du dataset PlantVillage :

- **version segmentée** : chaque image contient une feuille segmentée
  (dossier typiquement structuré en sous‑dossiers `Species___Disease`).
- **version couleur non segmentée** : images couleur brutes avant segmentation.

Organisation typique :

- `dataset/plantvillage/data/plantvillage dataset/segmented`  
  Dossiers `Espèce___Maladie` (ex. `Apple___Apple_scab`, `Cherry_(including_sour)___healthy`).
- `dataset/plantvillage/data/plantvillage dataset/color`  
  Version couleur non segmentée.


### 1.2 Nettoyage et préparation des données

- **Nettoyage du dataset couleur**  
  Script : `dataset/plantvillage/clean_color_dataset.py`

  Rôle :
  - détecter les **doublons exacts** (hash SHA‑256) et ne garder qu’une image par groupe ;
  - filtrer les **images floues** via la variance du Laplacien (z‑score) ;
  - produire un **CSV de keep‑list** listant uniquement les images conservées :  
    `dataset/plantvillage/csv/deep_learning_clean_plantvillage_color.csv`.

- **Construction d’un split propre train/val/test** pour la classification d’images  
  Script : `scripts/prepare_clean_split.py`

  Entrées :
  - CSV de keep‑list avec une colonne `path` (et éventuellement `action=keep`).

  Sorties :
  - arborescence `dataset/plantvillage/clean_split/` :
    - `train/<classe>/*.jpg`
    - `val/<classe>/*.jpg`
    - `test/<classe>/*.jpg`
  - `class_index.json`, `splits.json`, `README.md` locaux au split.


## 2. Pipeline Machine Learning (features tabulaires)

### 2.1 Extraction de descripteurs d’images

Les descripteurs sont calculés à partir d’un CSV listant les images (chemin `filepath`) :

- Module : `src/data_loader/feature_extraction.py`
- Fonctions utilitaires : `src/helpers/helpers.py`

Types de features extraites (handcrafted + shape) :

- **Moments de Hu** (forme globale de la feuille).
- **HOG** (Histogram of Oriented Gradients, contours/texture).
- **Énergie fréquentielle FFT** (basse/moyenne/haute fréquence).
- **Texture GLCM** (contrast, energy, homogeneity, dissimilarity, correlation).
- **Statistiques couleur** (moyennes/écarts‑types en RGB et HSV).
- **Descripteurs de forme avancés** (aire, périmètre, circularité, compacité, fractal dimension, etc.).

Les CSV consolidés importants sont dans `dataset/plantvillage/csv/`, en particulier :

- `combined_data_plantvillage.csv` (métadonnées de base).
- `feature_combined_data_plantvillage.csv` (avec features extraites).
- `clean_data_plantvillage_segmented_all_with_features.csv` (dataset final propre pour les modèles ML).


### 2.2 Modèles ML supervisés

Les principaux scripts se trouvent dans `Machine_Learning/` :

- `Log_reg.py`  
  - Régression logistique multinomiale sur les features.  
  - Pipeline : `RobustScaler` → `LogisticRegression`.
  - CV stratifiée, grid‑search sur `C` et `class_weight`,
    métriques : **balanced accuracy**, **macro‑F1**.
  - Sauvegarde des rapports, matrices de confusion (CSV + HTML interactif) et diagnostics.

- `Xtra_trees.py`  
  - Modèle **ExtraTreesClassifier** avec robust scaling.
  - Grid‑search sur le nombre d’arbres et le `class_weight`.
  - Même schéma de validation, bootstrap d’IC, diagnostics détaillés.

Les résultats sont stockés sous :

- `results/Machine_Learning/extra_trees_baseline/`  
- `results/Machine_Learning/logreg_baseline/`


### 2.3 Interprétabilité ML avec SHAP

- Script : `features_engineering/analyse_shap_meilleur_modele.py`

Rôle :

- réentraîner le **meilleur modèle ML** (par défaut un RandomForest rapide) sur
  `clean_data_plantvillage_segmented_all_with_features.csv` ;
- tirer un échantillon stratifié d’images ;
- calculer les **valeurs SHAP** ;
- générer 3 figures pour le rapport :
  - importance globale des features ;
  - summary plot détaillé ;
  - top features par classe (heatmap) ;
- produire un **rapport JSON** de synthèse.

Sorties : `figures/shap_analysis/` et `figures/shap_analysis/shap_analysis_report.json`.


## 3. Pipeline Deep Learning

Les architectures DL sont basées sur **EfficientNetV2S** pré‑entraîné (ImageNet) et
utilisent **tf_keras** (tf-nightly + tf-keras-nightly). Les scripts principaux sont dans `Deep_Learning/`.

### 3.1 Mono‑tâche : species / health / disease

- Script : `Deep_Learning/archi1_mono_tache.py`

Tâches supportées (`--task`) :

- `species` : prédiction de l’**espèce** (14 classes).  
- `health` : **binaire** healthy vs diseased.  
- `disease` : maladie uniquement sur les échantillons malades (healthy exclu).

Caractéristiques :

- pipeline `tf.data` depuis les chemins d’images (augmentations légères) ;
- split stratifié train/val/test (option : réutiliser des splits JSON) ;
- backbone EfficientNetV2S gelé puis **fine‑tuning** partiel ;
- optimisation AdamW/Adam avec label smoothing ;
- callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint ;
- rapport complet : courbes, matrice de confusion, classification report, sanity grid.


### 3.2 Mono‑objectif, 2 têtes indépendantes

- Script : `Deep_Learning/archi2_mono_tache_2_tetes.py`

Deux tâches lancées séparément :

- `species` : 14 espèces.  
- `disease_all` : 21 classes (20 maladies + healthy).  

L’architecture sous‑jacente reste un **EfficientNetV2S** finement ajusté pour chaque tâche.


### 3.3 Mono‑objectif, 1 tête unique (35 classes)

- Script : `Deep_Learning/archi3_mono_tache_1_tete.py`

Une seule tête softmax sur **35 classes combinées** `Espèce_État` :

- ex. `Apple_healthy`, `Apple_apple_scab`, `Tomato_late_blight`, etc.

Le script :

- dérive ensuite des **rapports species** et **disease global** à partir des 35 classes prédictes ;
- génère courbes, matrices de confusion, rapports de classification combinés.


### 3.4 Architecture 9 multi‑tâches : Species + Health → Disease

- Script : `Deep_Learning/archi9_multi_tache_species_health_to_disease.py`

Principe :

- tête principale **species** (14 classes) ;
- tête auxiliaire interne **health** (2 classes, non exposée, utilisée comme feature) ;
- tête **disease** qui consomme :
  - les features du backbone,
  - les probabilités complètes de species,
  - la probabilité d’être malade issue de la tête health.

Particularités :

- échantillons healthy **masqués** pour la perte disease ;
- callback dédié pour suivre macro‑F1 species et disease uniquement sur les malades ;
- rapport Markdown complet + figures et fichiers JSON de synthèse.


### 3.5 Embeddings Keras + SVM (Archi 5)

- `Deep_Learning/archi5_export_embeddings_keras.py` :
  - extrait des **embeddings de backbone** (mobilenetv3 ou EfficientNetV2S) sur des splits train/val/test ;
  - produit `X_train.npy`, `X_val.npy`, `X_test.npy` et les labels associés.

- `Deep_Learning/archi5_train_svm_from_embeddings.py` :
  - entraîne des SVM (species, health, disease global/per‑species) sur ces embeddings ;
  - sauvegarde modèles, matrices de confusion, métriques détaillées.


### 3.6 Baseline YOLOv8 en classification

- Script : `scripts/train_yolov8_cls.py`

Rôle :

- entraîner un modèle **YOLOv8‑cls** sur le dossier `clean_split` produit plus haut ;
- sauver les artefacts sous `outputs/yolov8_cls/<run_name>`.


### 3.7 Évaluation commune Keras / YOLOv8

- Script : `scripts/evaluate_cls.py`

Permet d’évaluer :

- un modèle **Keras** sauvegardé (`.keras`) à partir de listes de chemins (`splits.json`) ;
- un modèle **YOLOv8‑cls** à partir d’un répertoire `train/val/test`.

Sorties :

- `per_class_metrics.csv` (précision, rappel, F1, support) ;
- `confusion_matrix.png` ;
- `summary.json` (accuracy globale, macro‑F1, #classes, #échantillons).


### 3.8 Interprétabilité DL par Grad‑CAM

- Script : `Deep_Learning/Interpretability/gradcam_quickstart_arch9.py`

Fonctionnalités :

- charge un modèle Archi 9 pré‑entraîné (`best_model.keras`) ;
- sélectionne automatiquement la tête disease ;
- génère **original, heatmap Grad‑CAM et overlay** pour une image fournie (ou une image par défaut du dataset) ;
- sauvegarde les résultats dans un dossier `comparisons/interpretability/...`.


## 4. Organisation du dépôt

Principaux dossiers :

- `dataset/plantvillage/` : données brutes, scripts de nettoyage et CSV intermédiaires.
- `src/` : code réutilisable (helpers, extraction de features, data loader générique).
- `Machine_Learning/` : scripts d’entraînement ML tabulaire (LogReg, ExtraTrees) + diagnostics.
- `Deep_Learning/` : architectures Keras multi/mono‑tâches, export d’embeddings, interprétabilité.
- `features_engineering/` : analyses spécifiques (SHAP, analyse de segments, etc.).
- `notebooks/` : notebooks d’exploration et de prototypage.
- `results/` : résultats structurés des expériences ML et DL.
- `figures/` : figures prêtes pour le rapport (SHAP, courbes, matrices de confusion…).
- `scripts/` : scripts d’orchestration (préparation des splits, entraînement YOLO, évaluation commune).


## 5. Installation de l’environnement

Le dépôt fournit un fichier **conda** : `conda_env.yml`.

Exemple (à adapter au nom exact d’environnement indiqué dans le YAML) :

```bash
conda env create -f conda_env.yml
conda activate <nom_env>
```

Les principales dépendances sont :

- Python scientifique : `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly` ;
- Vision et images : `opencv-python`, `Pillow`, `scikit-image` ;
- Deep Learning : `tensorflow` / `tf_keras`, `tensorflow-addons` (éventuellement),
  GPU/mixed‑precision selon la machine ;
- Interprétabilité : `shap` ;
- YOLO : `ultralytics`.


## 6. Reproduire les expériences (exemples)

Les chemins exacts dépendent de l’endroit où vous avez téléchargé PlantVillage.
Les commandes ci‑dessous sont **des exemples** : adaptez `--root` / `--data_root` à votre machine.

### 6.1 Nettoyage du dataset couleur et génération du CSV de keep‑list

```bash
python -u dataset/plantvillage/clean_color_dataset.py \
  --root "/chemin/vers/plantvillage dataset/color" \
  --z-threshold 5 \
  --csv "dataset/plantvillage/csv/deep_learning_clean_plantvillage_color.csv"
```

### 6.2 Création d’un split propre train/val/test

```bash
python -u scripts/prepare_clean_split.py \
  --keep_list_csv dataset/plantvillage/csv/deep_learning_clean_plantvillage_color.csv \
  --out dataset/plantvillage/clean_split \
  --seed 42 --train 0.7 --val 0.15 --test 0.15 --link symlink
```

### 6.3 Entraînement des modèles ML tabulaires

- Régression logistique :

```bash
python Machine_Learning/Log_reg.py
```

- Extra Trees :

```bash
python Machine_Learning/Xtra_trees.py
```

### 6.4 Entraînement d’une architecture DL mono‑tâche (ex. species)

```bash
python Deep_Learning/archi1_mono_tache.py \
  --task species \
  --data_root "/chemin/vers/plantvillage dataset/segmented" \
  --output_dir results/Deep_Learning/archi1_outputs_mono_species_effv2s_256_color_split \
  --epochs 60 --batch_size 64
```

### 6.5 YOLOv8 classification sur le split propre

```bash
python -u scripts/train_yolov8_cls.py \
  --split_root dataset/plantvillage/clean_split \
  --model yolov8s-cls.pt --epochs 5 --batch 32 --imgsz 256 --name exp_s
```

### 6.6 Évaluation d’un modèle (Keras ou YOLOv8)

Keras :

```bash
python -u scripts/evaluate_cls.py --framework keras \
  --weights <chemin_vers_best_model.keras> \
  --splits <chemin_vers_splits.json> \
  --out <dossier_de_sortie>
```

YOLOv8 :

```bash
python -u scripts/evaluate_cls.py --framework yolov8 \
  --weights outputs/yolov8_cls/exp_s/weights/best.pt \
  --split_root dataset/plantvillage/clean_split --split test \
  --out outputs/yolov8_cls/exp_s/eval_test
```

### 6.7 Interprétabilité

- SHAP sur le meilleur modèle ML :

```bash
python features_engineering/analyse_shap_meilleur_modele.py
```

- Grad‑CAM sur un modèle Archi 9 :

```bash
python Deep_Learning/Interpretability/gradcam_quickstart_arch9.py --image /chemin/vers/une_image.jpg
```


## 7. Résultats et rapport

Les principaux artefacts sont organisés sous :

- `results/` : métriques, rapports, matrices de confusion, diagnostics ML/DL.
- `figures/` : figures prêtes pour intégration dans un rapport scientifique.
- `results/Deep_Learning/.../report.md` : résumés par architecture.
- `results/Machine_Learning/...` : diagnostics complets (CV, bootstrap, courbes de validation).

Ce README sert de **vue d’ensemble** du projet :

- du nettoyage des données PlantVillage,
- à la comparaison ML vs Deep Learning,
- jusqu’à l’interprétabilité des modèles (SHAP, Grad‑CAM).
