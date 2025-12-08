# Approche Single-Model - Classification Directe

## 🎯 Concept

**UN SEUL modèle** EfficientNetV2B1 qui prédit directement les **38 classes finales** (espèce___maladie).

### Différences avec l'approche Cascade

| Aspect | Cascade (2 modèles) | Single Model (1 modèle) |
|--------|---------------------|-------------------------|
| **Architecture** | Espèce → Maladie (conditionnel) | Direct → 38 classes |
| **Entraînement** | 2 modèles séparés | 1 modèle unique |
| **Inférence** | 2 passes forward | 1 passe forward |
| **Classes "healthy"** | Traitement spécial | Classe normale |
| **Erreurs** | Propagation espèce → maladie | Pas de propagation |
| **Complexité** | Plus complexe | Plus simple |

## 🏗️ Architecture

```
Input Image (256×256×3)
         ↓
EfficientNetV2B1 (ImageNet)
         ↓
GlobalAveragePooling
         ↓
Dropout(0.5)
         ↓
Dense(512, relu) + L2
         ↓
Dropout(0.3)
         ↓
Dense(38, softmax)
         ↓
38 classes finales
```

### Classes finales (38)

```
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy          ← 'healthy' est une classe normale!
...
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___healthy         ← 'healthy' est une classe normale!
...
```

## 📊 Entraînement

### Configuration

- **Backbone**: EfficientNetV2B1 (~8M params)
- **Split**: Stratifié avec SEED=42
  - Train: ~70%
  - Val: ~15%
  - Test: ~15%
- **Batch size**: 32
- **Image size**: 256×256

### Stratégie de Fine-tuning (2 phases)

#### Phase 1: Head Only (15 époques)
- Backbone EfficientNetV2B1 **gelé**
- Entraînement de la tête uniquement (Dense layers)
- Learning rate: 1e-3
- Label smoothing: 0.1

#### Phase 2: Fine-tuning (40 époques)
- Dégel des **80 dernières couches** du backbone
- Learning rate: 1e-4 (10x plus petit)
- Label smoothing: 0.1
- Early stopping: patience=7
- ReduceLROnPlateau: patience=3

**Total: 55 époques**

## 📈 Visualisations générées automatiquement

### 1. Graphiques d'entraînement (`training_curves.png`)

Grille 2×2 avec:
- **Loss vs Epochs** (train + val)
- **Accuracy vs Epochs** (train + val)
- **AUC vs Epochs** (train + val)
- **Learning Rate vs Epochs** (log scale)

### 2. Matrices de confusion (Heatmaps)

#### `confusion_matrix_full.png`
- Matrice 38×38 complète
- Couleur: Jaune → Orange → Rouge
- Diagonale = prédictions correctes

#### `confusion_matrix_normalized.png`
- Normalisée en % par classe vraie
- Couleur: Rouge (erreurs) → Vert (correct)
- Montre la proportion d'erreurs

#### `confusion_matrix_errors.png`
- **Diagonale exclue** (= 0)
- Montre UNIQUEMENT les erreurs
- Très sparse si bon modèle!

### 3. Rapport détaillé

- `classification_report.csv`: Precision, Recall, F1 pour chaque classe
- `top_confusions.csv`: Top confusions détaillées
- `results.json`: Métriques globales (accuracy, F1 macro, F1 weighted)

## 🎯 Métriques

### Métriques globales calculées

```json
{
  "accuracy": 0.XXXX,
  "f1_macro": 0.XXXX,      // Toutes classes égales
  "f1_weighted": 0.XXXX,   // Pondéré par support (PRINCIPAL)
  "num_classes": 38,
  "num_test_samples": 8146
}
```

### Comparaison attendue avec Cascade

**Avantages du Single-Model:**
- ✅ Plus simple (1 modèle vs 2)
- ✅ Plus rapide (1 inférence vs 2)
- ✅ Pas de propagation d'erreur
- ✅ 'healthy' traité uniformément

**Inconvénients potentiels:**
- ❌ Moins de contraintes structurelles (espèce → maladie)
- ❌ Plus de classes à apprendre simultanément (38 vs 14+21)

## 🚀 Utilisation

### Entraînement

```bash
python train_single_model_efficientnet.py \
  --epochs_phase1 15 \
  --epochs_phase2 40 \
  --batch_size 32 \
  --patience 7
```

### Monitoring

```bash
bash monitor_single_model.sh
```

### Résultats

Tous les fichiers dans `outputs/single_model_efficientnet/`:
```
single_model_efficientnet.keras     # Modèle final
training_history.csv                # Historique complet
training_curves.png                 # Graphiques 4-en-1
confusion_matrix_full.png           # Heatmap complète
confusion_matrix_normalized.png     # Heatmap %
confusion_matrix_errors.png         # Erreurs only
classification_report.csv           # Détails par classe
top_confusions.csv                  # Top erreurs
results.json                        # Métriques globales
class_mapping.json                  # Mapping classes
```

## 📝 Notes importantes

1. **SEED=42** fixé pour reproducibilité
2. **Mixed precision** activée (FP16) pour accélération
3. **Label smoothing** (0.1) pour régularisation
4. **Early stopping** pour éviter overfitting
5. **Stratified split** pour garder distribution des classes

## 🔬 À analyser après entraînement

1. **Comparer avec Cascade**:
   - Accuracy, F1 macro, F1 weighted
   - Vitesse d'inférence
   - Complexité du modèle

2. **Analyser les confusions**:
   - Erreurs intra-espèce (même espèce, maladie différente)?
   - Erreurs inter-espèces (espèce différente)?
   - 'healthy' confondu avec quelles maladies?

3. **Analyser la convergence**:
   - Overfitting?
   - Sous-apprentissage?
   - Optimal stopping?

## 💡 Améliorations possibles

- Test avec EfficientNetV2B2/B3 (plus gros)
- Data augmentation (rotation, flip, color jitter)
- Test avec d'autres backbones (ConvNeXt, SwinTransformer)
- Ensembling avec Cascade
- Focal loss pour classes déséquilibrées
