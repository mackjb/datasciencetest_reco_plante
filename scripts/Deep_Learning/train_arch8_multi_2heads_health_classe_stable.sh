#!/bin/bash
# Script pour entraîner le modèle multi-tâches 2 têtes (species + disease_all avec health comme classe)
# Architecture simplifiée : healthy = 21ème classe de disease_all

echo "🚀 Entraînement multi-tâches 2 têtes (species + disease_all)"
echo "============================================================="
echo ""
echo "Architecture :"
echo "  - Species      : 14 classes"
echo "  - Disease_all  : 21 classes (20 maladies + healthy)"
echo ""
echo "Paramètres stabilisés :"
echo "  - Learning rate fine-tuning : 1e-4 → 1e-5 (10x plus petit)"
echo "  - Learning rate initial     : 1e-3 → 5e-4"
echo "  - Couches dégelées         : 50 → 30"
echo "  - Gradient clipping        : DÉSACTIVÉ (gradient_clip=0)"
echo "  - Loss weight species      : 1.0"
echo "  - Loss weight disease_all  : 1.0 (équilibré)"
echo ""
echo "============================================================="
echo ""

/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/multi_tache_2_tetes_health_est_une_classe.py \
  --data_root "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color" \
  --output_dir /workspaces/app/outputs_multi_2heads_health_classe_stable \
  --splits_file /workspaces/app/splits/pv_color_splits.json \
  --epochs 60 \
  --batch_size 64 \
  --img_size 256 256 \
  --initial_lr 5e-4 \
  --ft_lr 1e-5 \
  --fine_tune_at 30 \
  --gradient_clip 0 \
  --loss_w_species 1.0 \
  --loss_w_disease 1.0 \
  --weight_decay 1e-4 \
  --label_smoothing_species 0.1 \
  --label_smoothing_disease 0.1 \
  --no_sanity_grid

echo ""
echo "✅ Entraînement terminé !"
echo "📊 Résultats disponibles dans : outputs_multi_2heads_health_classe_stable/"
echo ""
echo "Pour visualiser les courbes macro F1 :"
echo "  python plot_macro_f1_multi.py outputs_multi_2heads_health_classe_stable"
echo ""
echo "Comparaison avec le modèle 3 têtes :"
echo "  - Avantages : Architecture plus simple, pas de filtrage healthy"
echo "  - Disease_all : Toutes les images entraînent cette tête (y compris healthy)"
