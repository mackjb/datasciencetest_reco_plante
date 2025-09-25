import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Désactiver les logs verbeux
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
BASE_DIR = "/workspaces/datasciencetest_reco_plante"
DATA_DIR = os.path.join(BASE_DIR, "dataset/plantvillage/data/plantvillage_5images/segmented")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results/models/resnet_final.h5")

# Créer les dossiers nécessaires
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Paramètres
BATCH_SIZE = 4
EPOCHS = 5
IMG_SIZE = (128, 128)

print("=== Chargement des données ===")
# Chargement des données avec augmentation minimale
train_datagen = ImageDataGenerator(
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
base_model = ResNet50(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Afficher un résumé du modèle
model.summary()

# Entraînement simplifié
print("\n=== Début de l'entraînement ===")
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = val_generator.samples // BATCH_SIZE

for epoch in range(EPOCHS):
    print(f"\nEpoque {epoch+1}/{EPOCHS}")
    
    # Entraînement
    print("Entraînement:", end=" ")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=1,
        verbose=1
    )
    
    # Sauvegarde après chaque époque
    model.save(MODEL_SAVE_PATH)
    print(f"Modèle sauvegardé: {MODEL_SAVE_PATH}")

# Évaluation finale
print("\n=== Évaluation finale ===")
loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"\nRésultats finaux:")
print(f"- Précision sur l'ensemble de validation: {accuracy*100:.2f}%")
print(f"- Perte sur l'ensemble de validation: {loss:.4f}")
print(f"\nModèle sauvegardé dans: {MODEL_SAVE_PATH}")
