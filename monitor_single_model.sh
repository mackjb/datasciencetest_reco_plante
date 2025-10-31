#!/bin/bash
# Script de monitoring pour le modèle unique EfficientNetV2

LOG_FILE="outputs/single_model_efficientnet/training.log"

echo "=================================================="
echo "MONITORING - MODÈLE UNIQUE EFFICIENTNETV2B1"
echo "Classification directe 38 classes (espèce___maladie)"
echo "=================================================="
echo ""

# Vérifier si le processus tourne
PID=$(ps aux | grep "train_single_model_efficientnet" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Processus non actif"
else
    echo "✅ Processus actif (PID: $PID)"
    CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep $PID | grep -v grep | awk '{print $4}')
    echo "   CPU: ${CPU}%, MEM: ${MEM}%"
fi

echo ""
echo "=================================================="
echo "PROGRESSION"
echo "=================================================="

# Dernières époques
echo ""
echo "📊 Dernières époques:"
tail -1000 "$LOG_FILE" | grep -E "Epoch [0-9]+/" | tail -10

echo ""
echo "=================================================="
echo "MÉTRIQUES RÉCENTES"
echo "=================================================="

# Dernières métriques
echo ""
echo "🎯 Dernières métriques:"
tail -300 "$LOG_FILE" | grep -E "val_accuracy|Val Acc:|Val F1:" | tail -8

echo ""
echo "=================================================="
echo "PHASE ACTUELLE"
echo "=================================================="
tail -100 "$LOG_FILE" | grep -E "PHASE|Fine-tuning" | tail -2

echo ""
echo "=================================================="
echo "DERNIÈRES LIGNES DU LOG"
echo "=================================================="
tail -15 "$LOG_FILE" 2>/dev/null | head -12

echo ""
echo "=================================================="
echo "Pour voir le log complet: tail -f $LOG_FILE"
echo "=================================================="
