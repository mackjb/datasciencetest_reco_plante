#!/bin/bash
# Script pour entraîner l'Architecture 2 : Mono-tâche 2 têtes (séparés)
# Entraînement séquentiel de 2 modèles spécialisés (Species et DiseaseAll)

echo "🚀 Entraînement Architecture 2 : Mono-tâche (2 modèles séparés)"
echo "============================================================"
echo ""
echo "Description :"
echo "  Cette architecture entraîne 2 modèles distincts :"
echo "  1. Modèle Species : Classification des 14 espèces"
echo "  2. Modèle DiseaseAll : Classification des 21 classes (20 maladies + healthy)"
echo ""
echo "Différence avec Arch 1 :"
echo "  - Pas de modèle Health binaire séparé."
echo "  - Le modèle DiseaseAll inclut 'healthy' comme une classe à part entière."
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
echo "🌿 [1/2] Démarrage tâche SPECIES..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/mono_tache_2_tetes.py \
  --task species \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_mono_disease_2heads_effv2s_256_color_split \
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

# 2. Entraînement Tâche DiseaseAll
echo "🦠 [2/2] Démarrage tâche DISEASE_ALL (21 classes)..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/mono_tache_2_tetes.py \
  --task disease_all \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_mono_disease_2heads_effv2s_256_color_split_disease_all \
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

echo "✅ Tâche DISEASE_ALL terminée."
echo ""

echo "============================================================"
echo "🎉 Tous les entraînements de l'Architecture 2 sont terminés !"
echo "📊 Résultats disponibles dans :"
echo "  - outputs_mono_2heads_species/"
echo "  - outputs_mono_2heads_disease_all/"
echo ""
