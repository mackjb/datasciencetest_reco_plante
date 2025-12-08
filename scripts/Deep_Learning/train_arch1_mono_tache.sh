#!/bin/bash
# Script pour entraîner l'Architecture 1 : Mono-tâche 3 modèles séparés
# Entraînement séquentiel de 3 modèles spécialisés

echo "🚀 Entraînement Architecture 1 : Mono-tâche (3 modèles séparés)"
echo "============================================================"
echo ""
echo "Description :"
echo "  Cette architecture entraîne 3 modèles distincts :"
echo "  1. Modèle Species : Classification des 14 espèces"
echo "  2. Modèle Health  : Classification binaire (Healthy vs Diseased)"
echo "  3. Modèle Disease : Classification des 20 maladies (sur plantes malades uniquement)"
echo ""
echo "Paramètres communs :"
echo "  - Backbone          : EfficientNetV2S"
echo "  - Initial LR        : 1e-3"
echo "  - Fine-tune LR      : 1e-4"
echo "  - Fine-tune at      : 50 layers"
echo "  - Epochs            : 60"
echo ""
echo "============================================================"
echo ""

# 1. Entraînement Tâche Species
echo "🌿 [1/3] Démarrage tâche SPECIES..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/mono_tache.py \
  --task species \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_mono_species_effv2s_256_color_split \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 1e-3 \
  --ft_lr 1e-4 \
  --fine_tune_at 50 \
  --weight_decay 1e-4 \
  --label_smoothing 0.1 \
  --no_sanity_grid

echo "✅ Tâche SPECIES terminée."
echo ""

# 2. Entraînement Tâche Health
echo "🏥 [2/3] Démarrage tâche HEALTH (Healthy vs Diseased)..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/mono_tache.py \
  --task health \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_mono_health_effv2s_256_color_split \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 1e-3 \
  --ft_lr 1e-4 \
  --fine_tune_at 50 \
  --weight_decay 1e-4 \
  --label_smoothing 0.1 \
  --no_sanity_grid

echo "✅ Tâche HEALTH terminée."
echo ""

# 3. Entraînement Tâche Disease
echo "🦠 [3/3] Démarrage tâche DISEASE (Classification des maladies)..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/mono_tache.py \
  --task disease \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_mono_disease_effv2s_256_color_split \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 1e-3 \
  --ft_lr 1e-4 \
  --fine_tune_at 50 \
  --weight_decay 1e-4 \
  --label_smoothing 0.1 \
  --no_sanity_grid

echo "✅ Tâche DISEASE terminée."
echo ""

echo "============================================================"
echo "🎉 Tous les entraînements de l'Architecture 1 sont terminés !"
echo "📊 Résultats disponibles dans :"
echo "  - outputs_mono_species/"
echo "  - outputs_mono_health/"
echo "  - outputs_mono_disease/"
echo ""
