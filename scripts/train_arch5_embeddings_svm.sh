#!/bin/bash
# Script pour entraîner l'Architecture 5 :
#  - Étape 1 : Extraction des embeddings Keras (EfficientNetV2S multi-tâche)
#  - Étape 2 : Entraînement des SVM sur ces embeddings

echo "🚀 Entraînement Architecture 5 : Embeddings Keras + SVM"
echo "======================================================="
echo ""
echo "Étape 1 : Export des embeddings à partir du modèle multi-tâche EfficientNetV2S"
echo "Étape 2 : Entraînement des SVM (species, health, disease) sur ces embeddings"
echo ""

# ------------------------------------------------------------------
# Chemins et paramètres communs
# ------------------------------------------------------------------
DATA_ROOT="/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color"
SPLITS_FILE="/workspaces/app/splits/pv_color_splits.json"

# Dossier du meilleur modèle multi-tâche EfficientNetV2S
MT_DIR="/workspaces/app/outputs_plantvillage_mt_effv2s_frozen_gpu"
FROM_KERAS_MODEL="${MT_DIR}/best_multitask.keras"

# Dossiers de sortie pour archi 5
EMB_OUT="/workspaces/app/outputs_embeddings_arch5_same_splits"
SVM_OUT="/workspaces/app/svm_results_arch5_same_splits"

IMG_SIZE=256
BATCH=64
PCA_DIM=0

echo "Données      : ${DATA_ROOT}"
echo "Splits       : ${SPLITS_FILE}"
echo "Modèle Keras : ${FROM_KERAS_MODEL}"
echo "Embeddings   : ${EMB_OUT}"
echo "Résultats SVM: ${SVM_OUT}"
echo ""

# ------------------------------------------------------------------
# Étape 1 : Export des embeddings Keras
# ------------------------------------------------------------------
echo "📦 [1/2] Export des embeddings Keras..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/dl/export_embeddings_keras.py \
  --data_root "${DATA_ROOT}" \
  --img_size ${IMG_SIZE} \
  --batch ${BATCH} \
  --outdir "${EMB_OUT}" \
  --backbone efficientnetv2s \
  --splits_file "${SPLITS_FILE}" \
  --labels_dir "${MT_DIR}" \
  --from_keras_model "${FROM_KERAS_MODEL}"

echo "✅ Export des embeddings terminé."
echo ""

# ------------------------------------------------------------------
# Étape 2 : Entraînement des SVM à partir des embeddings
# ------------------------------------------------------------------
echo "🤖 [2/2] Entraînement des SVM sur les embeddings..."
/workspaces/app/micromamba run -n tf_gpu_new python /workspaces/app/dl/train_svm_from_embeddings.py \
  --emb_dir "${EMB_OUT}" \
  --outdir "${SVM_OUT}" \
  --pca_dim ${PCA_DIM} \
  --grid \
  --n_jobs -1 \
  --cv 3 \
  --scoring auto

echo ""
echo "======================================================="
echo "🎉 Architecture 5 terminée !"
echo "📂 Embeddings : ${EMB_OUT}"
echo "📊 Résultats SVM : ${SVM_OUT}"
echo "======================================================="
