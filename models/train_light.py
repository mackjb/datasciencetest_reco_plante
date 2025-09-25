import os
import sys
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Désactiver les logs verbeux
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
BASE_DIR = "/workspaces/datasciencetest_reco_plante"
DATA_DIR = os.path.join(BASE_DIR, "dataset/plantvillage/data/plantvillage_5images/segmented")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results/models/resnet_light.h5")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Paramètres réduits
BATCH_SIZE = 4  # Réduit pour économiser de la mémoire
EPOCHS = 5      # Moins d'époques pour un test rapide
IMG_SIZE = (128, 128)  # Images plus petites

# Barre de progression personnalisée
class ProgressBar:
    def __init__(self, total, desc=''):
        self.pbar = tqdm(total=total, desc=desc, ncols=100)
        self.total = total
    
    def update(self, n=1, logs=None):
        self.pbar.update(n)
        if logs:
            self.pbar.set_postfix({
                'loss': f"{logs.get('loss', 0):.4f}",
                'acc': f"{logs.get('accuracy', 0)*100:.1f}%"
            })
    
    def close(self):
        self.pbar.close()

print("=== Chargement des données ===")
# Chargement des données avec augmentation minimale
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("\n=== Construction du modèle ResNet50 ===")
# Chargement de ResNet50
base_model = tf.keras.applications.ResNet50(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement avec barre de progression
print("\n=== Début de l'entraînement ===")
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = val_generator.samples // BATCH_SIZE

for epoch in range(EPOCHS):
    print(f"\nEpoque {epoch+1}/{EPOCHS}")
    
    # Entraînement
    train_pbar = ProgressBar(steps_per_epoch, 'Entraînement')
    for step in range(steps_per_epoch):
        x, y = next(train_generator)
        metrics = model.train_on_batch(x, y, return_dict=True)
        train_pbar.update(1, metrics)
    train_pbar.close()
    
    # Validation
    val_loss, val_acc = 0, 0
    val_pbar = tqdm(total=validation_steps, desc='Validation', ncols=100)
    for _ in range(validation_steps):
        x, y = next(val_generator)
        loss, acc = model.test_on_batch(x, y)
        val_loss += loss
        val_acc += acc
        val_pbar.update(1)
    val_pbar.close()
    
    # Affichage des métriques
    val_loss /= validation_steps
    val_acc /= validation_steps
    print(f"  - val_loss: {val_loss:.4f} - val_acc: {val_acc*100:.2f}%")

# Sauvegarde finale
model.save(MODEL_SAVE_PATH)
print(f"\nModèle sauvegardé dans {MODEL_SAVE_PATH}")

# Évaluation finale
print("\n=== Évaluation finale ===")
loss, accuracy = model.evaluate(val_generator, verbose=0)
print(f"Précision sur l'ensemble de validation: {accuracy*100:.2f}%")
