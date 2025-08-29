# ExtraTrees PlantVillage v2 — Mémo de synthèse (2025-08-22)

Auteur: Cascade • Projet: PlantVillage • Module: `models/extratrees_plantvillage_v2/`

## 1) Objectif
Construire un pipeline ExtraTrees robuste et traçable sur PlantVillage, avec bonnes pratiques ML: prétraitement soigné, gestion du déséquilibre, CV stratifiée, tuning, métriques pertinentes, interprétabilité, contrôle des ressources et sauvegarde exhaustive des artefacts.

## 2) Données & préparation
- Source: `dataset/plantvillage/csv/clean_with_features_data_plantvillage_segmented_all.csv`
- Post-nettoyage: 6 871 lignes • 42 features numériques • 0 catégorielle • 14 classes (`species`).
- Nettoyage: drop NA sur `species`, exclusion `is_na==True` et `is_duplicate_after_first==True` si présents.

## 3) Pipeline & justifications
- Numériques: `SimpleImputer(median)` + winsorisation 1–99% (réduit l’influence des extrêmes sans normaliser la distribution).
- Catégorielles: `OneHotEncoder(handle_unknown="ignore")` (pas utilisé ici car 0 cat, mais pipeline compatible).
- Pas de scaling: arbres invariants aux échelles → inutile et potentiellement bruité.
- Modèle: `ExtraTreesClassifier` (rapide, robuste, peu sensible au bruit, importances disponibles).

## 4) Déséquilibre de classes
- Comparaison contrôlée: `class_weight ∈ {None, "balanced"}` vs `RandomOverSampler` (ROS).
- Ne pas cumuler class_weight et oversampling (évite double correction → sur-apprentissage possible).

## 5) Validation & métriques
- Split + CV 5-fold stratifiés (reproductibles).
- Scoring: `f1_macro` (métrique principale) + `balanced_accuracy` (complément).
  - Macro-F1: moyenne non pondérée des F1 par classe → adapté au déséquilibre multi-classes.
  - Micro-F1 ("global"): ≈ accuracy en multi-classes, dominé par majoritaires → non retenu en primaire.

## 6) Tuning
- Grille (24 candidats):
  - `n_estimators ∈ {200, 400}`, `max_depth=None`, `min_samples_split ∈ {2, 5}`, `min_samples_leaf ∈ {1, 2}`
  - `max_features="sqrt"`, `bootstrap=False`, `criterion="gini"`
  - `sampler ∈ {passthrough, ROS}`, `class_weight ∈ {None, "balanced"}` (sans cumul)
- `GridSearchCV` (StratifiedKFold(5), `refit="f1_macro"`, `n_jobs_outer=1`).

## 7) Résultats principaux
- Baseline CV (sans tuning):
  - mean test Balanced Acc ≈ 0.7662 • mean test F1-macro ≈ 0.7625
- Meilleurs hyperparamètres (CV refit macro-F1):
  ```json
  {"sampler": "passthrough", "model__class_weight": null, "model__n_estimators": 400,
   "model__max_features": "sqrt", "model__min_samples_split": 2, "model__min_samples_leaf": 1,
   "model__criterion": "gini", "model__bootstrap": false, "model__max_depth": null}
  ```
  - Meilleur score CV (f1_macro): 0.7625
- Test set:
  - Balanced Accuracy: 0.7676
  - F1-macro: 0.7659

## 8) Interprétabilité
- Importances (top): `std_B`, `mean_B`, `mean_H`, `std_G`, `mean_R`, `contrast`, `std_R`, `dissimilarite`, `phi2_distinction_elongation_forme`, `fft_entropy`.
- Fichiers: `results/models/extratrees_plantvillage_v2/feature_importances.csv` (+ figures si activées).
- SHAP: désactivé par défaut. Activer avec `ENABLE_SHAP=1` pour générer `results/.../shap/*.png`.

## 9) Artefacts & traçabilité
- Dossier: `results/models/extratrees_plantvillage_v2/`
  - `cv_baseline_scores.csv`, `gridsearch_results.csv`, `best_params.json`
  - `evaluation/{classification_report.txt, confusion_matrix.csv, predictions_test.csv}`
  - `feature_importances.csv`, `plots/{cv_results_enriched.csv, best_by_approach.csv, scatter_f1_vs_n_estimators.html}`
  - Registre JSONL de tous les essais: `grid_trials.jsonl`

## 10) Reproductibilité & ressources
- Commande:
  ```bash
  N_JOBS_ESTIMATOR=2 N_JOBS_OUTER=1 ENABLE_SHAP=0 \
  python -m models.extratrees_plantvillage_v2
  ```
- Parallélisme contrôlé pour éviter la surconcurrence CPU (threads internes ExtraTrees vs jobs externes Grid/CV).

## 11) Recommandations & pistes
- Étendre la grille: `criterion ∈ {"gini","entropy"}`, `ccp_alpha`, `min_impurity_decrease`.
- Tester d’autres samplers: RUS, SMOTE (attention aux features cat si présentes → SMOTENC).
- Analyser erreurs via `evaluation/predictions_test.csv` (confusions récurrentes par classes proches).
- Activer SHAP pour un diagnostic local/global (vérifier ressources).
- Valider robustesse avec seeds multiples et/ou CV stratifiée répétée.
