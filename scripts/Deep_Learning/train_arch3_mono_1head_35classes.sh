#!/bin/bash
# Script pour entraîner l'Architecture 3 : 1 tête unique (35 classes Species_State)
# Prédiction simultanée de l'espèce ET de l'état de santé/maladie

echo "🚀 Entraînement Architecture 3 : 1 tête unique (35 classes)"
echo "============================================================"
echo ""
echo "Architecture :"
echo "  - Backbone → Features"
echo "  - Features → Dense(35, softmax)"
echo "  - Format des classes : Espèce_État (ex: Apple_healthy, Tomato_late_blight)"
echo ""
echo "Paramètres :"
echo "  - Classes totales   : 35 (14 espèces × états de santé)"
echo "  - Initial LR        : 1e-3"
echo "  - Fine-tune LR      : 1e-4"
echo "  - Fine-tune at      : 50 layers"
echo "  - Gradient clip     : 1.0"
echo "  - Epochs            : 60"
echo ""
echo "============================================================"
echo ""

/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/mono_tache_1_tete.py \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_mono_1head_35classes \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 1e-3 \
  --ft_lr 1e-4 \
  --fine_tune_at 50 \
  --gradient_clip 1.0 \
  --weight_decay 1e-4 \
  --label_smoothing 0.1 \
  --no_sanity_grid

echo ""
echo "✅ Entraînement terminé !"
echo "📊 Résultats dans : outputs_mono_1head_35classes/"
echo ""
echo "Pour générer uniquement le rapport (si modèle existe) :"
echo "  bash train_mono_1head_35classes.sh --report_only"
