#!/bin/bash
# Script pour entraîner le modèle multi-tâches 3 têtes SANS fine-tuning
# Seules les têtes sont entraînées, le backbone reste gelé

echo "🚀 Entraînement multi-tâches 3 têtes (species, health, disease)"
echo "==============================================================="
echo ""
echo "⚠️  MODE: SANS FINE-TUNING"
echo "   - Backbone EfficientNetV2S : GELÉ"
echo "   - Entraînement : Têtes uniquement"
echo "   - Pas de déblocage de couches"
echo ""
echo "Paramètres :"
echo "  - Epochs               : 60"
echo "  - Learning rate        : 1e-3 (initial_lr uniquement)"
echo "  - Batch size           : 64"
echo "  - Image size           : 256x256"
echo "  - Loss weight species  : 1.0"
echo "  - Loss weight health   : 0.5"
echo "  - Loss weight disease  : 1.5"
echo ""
echo "Output : /workspaces/app/outputs_multi_effv2s_256_color_split_no_finetuning"
echo ""
echo "==============================================================="
echo ""

/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/multi_tache.py \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_multi_effv2s_256_color_split_no_finetuning \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 1e-3 \
  --loss_w_species 1.0 \
  --loss_w_health 0.5 \
  --loss_w_disease 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing_species 0.1 \
  --label_smoothing_disease 0.1 \
  --no_sanity_grid

echo ""
echo "✅ Entraînement terminé !"
echo "📊 Résultats disponibles dans : outputs_multi_effv2s_256_color_split_no_finetuning/"
echo ""
echo "Pour visualiser les courbes macro F1 :"
echo "  python plot_macro_f1_multi.py outputs_multi_effv2s_256_color_split_no_finetuning"
echo ""
echo "📝 Note : Ce modèle N'A PAS de fine-tuning"
echo "   - Plus rapide à entraîner"
echo "   - Performance potentiellement inférieure"
echo "   - Utile pour baseline ou tests rapides"
