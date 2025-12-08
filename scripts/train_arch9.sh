#!/bin/bash
# Script pour entraîner l'Architecture 9 : Species + Health → Disease
# La tête Disease profite des probabilités Species ET Health

echo "🚀 Entraînement Architecture 9 : Species + Health → Disease"
echo "============================================================"
echo ""
echo "Architecture :"
echo "  - Backbone → Features"
echo "  - Features → Species (Dense 14)"
echo "  - Features → Health auxiliaire (Dense 2, non exposé)"
echo "  - Features + Species probs + Health prob → Disease (Dense 20)"
echo ""
echo "Paramètres :"
echo "  - Initial LR    : 1e-3"
echo "  - Fine-tune LR  : 1e-4"
echo "  - Fine-tune at  : 50 layers"
echo "  - Gradient clip : 1.0"
echo "  - Epochs        : 60"
echo ""
echo "============================================================"
echo ""

/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/multi_tache_arch9_species_health_to_disease.py \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_arch9_species_health_to_disease \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 1e-3 \
  --ft_lr 1e-4 \
  --fine_tune_at 50 \
  --gradient_clip 1.0 \
  --loss_w_species 1.0 \
  --loss_w_disease 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing_species 0.1 \
  --label_smoothing_disease 0.1 \
  --no_sanity_grid

echo ""
echo "✅ Entraînement terminé !"
echo "📊 Résultats dans : outputs_arch9_species_health_to_disease/"
echo ""
echo "Pour générer uniquement le rapport (si modèle existe) :"
echo "  bash train_arch9.sh --report_only"
