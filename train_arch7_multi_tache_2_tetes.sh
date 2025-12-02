#!/bin/bash
# Script pour entraîner l'Architecture 7 : Multi-tâche 2 têtes (Species + Disease)
# Entraînement simultané avec masquage des échantillons sains pour la tête Disease

echo "🚀 Entraînement Architecture 7 : Multi-tâche 2 têtes (Species + Disease)"
echo "============================================================"
echo ""
echo "Description :"
echo "  - 2 têtes de sortie : Species (14 classes) et Disease (20 classes)"
echo "  - Utilise une tête 'Health' auxiliaire interne (cachée) pour aider la tête Disease"
echo "  - Masquage de perte : les échantillons 'healthy' ont un poids de 0 pour la perte Disease"
echo ""
echo "Paramètres :"
echo "  - Backbone          : EfficientNetV2S"
echo "  - Epochs            : 60"
echo "  - Loss weights      : Species=1.0, Disease=1.5"
echo "  - Batch size        : 64"
echo ""
echo "============================================================"
echo ""

# Entraînement
echo "🧠 Démarrage de l'entraînement..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/multi_tache_2_tetes.py \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_multi_2heads_effv2s_256_color_split \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 1e-3 \
  --ft_lr 1e-4 \
  --fine_tune_at 50 \
  --weight_decay 1e-4 \
  --label_smoothing_species 0.1 \
  --label_smoothing_disease 0.1 \
  --loss_w_species 1.0 \
  --loss_w_disease 1.5 \
  --no_sanity_grid

echo ""
echo "============================================================"
echo "🎉 Entraînement Architecture 7 terminé !"
echo "📊 Résultats disponibles dans : outputs_multi_2heads_arch7/"
echo ""
