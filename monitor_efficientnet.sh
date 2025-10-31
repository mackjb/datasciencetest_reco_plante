#!/bin/bash
# Script de monitoring pour l'entraînement EfficientNetV2

LOG_FILE="outputs/cascade_efficientnet/training.log"

echo "=================================================="
echo "MONITORING - CASCADE EFFICIENTNETV2"
echo "=================================================="
echo ""

# Vérifier si le processus tourne
PID=$(ps aux | grep "train_cascade_efficientnet" | grep -v grep | awk '{print $2}')
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

# Dernières métriques de validation
echo ""
echo "🎯 Dernières métriques Val:"
tail -200 "$LOG_FILE" | grep -E "val_accuracy|Val Acc:|Val F1:" | tail -8

echo ""
echo "=================================================="
echo "DERNIÈRES LIGNES DU LOG"
echo "=================================================="
tail -15 "$LOG_FILE"

echo ""
echo "=================================================="
echo "Pour voir le log complet: tail -f $LOG_FILE"
echo "=================================================="
