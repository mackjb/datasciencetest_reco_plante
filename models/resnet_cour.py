import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Chemins des données et modèles ---
BASE_DIR = "/workspaces/datasciencetest_reco_plante"
DATA_DIR = os.path.join(BASE_DIR, "dataset/plantvillage")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results/models/resnet_plant_disease.h5")

# --- Paramètres ---
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# --- Générateurs de données avec augmentation ---
def create_data_generators():
    """Crée les générateurs d'images pour l'entraînement et la validation"""
    # Vérifier si le dossier de données existe
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Le dossier de données {DATA_DIR} n'existe pas.")
    
    # Augmentation de données pour l'entraînement
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )

    # Pas d'augmentation pour la validation
    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=VALIDATION_SPLIT
    )

    # Générateur d'entraînement
    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=42
    )

    # Générateur de validation
    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    
    return train_gen, val_gen, train_gen.class_indices

def build_model(num_classes):
    """Construit le modèle ResNet50 avec une nouvelle tête"""
    # Charger ResNet50 pré-entraîné sur ImageNet
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    
    # Geler les couches de base
    base_model.trainable = False
    
    # Construire le modèle
    inputs = Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def train():
    """Entraîne le modèle"""
    # Créer les générateurs de données
    train_gen, val_gen, class_indices = create_data_generators()
    num_classes = len(class_indices)
    
    # Sauvegarder le mapping des classes
    class_mapping_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'class_indices.npy')
    np.save(class_mapping_path, class_indices)
    
    # Construire le modèle
    model = build_model(num_classes)
    
    # Compiler le modèle
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Entraînement
    print("Démarrage de l'entraînement...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, val_gen):
    """Évalue le modèle sur l'ensemble de validation"""
    print("\nÉvaluation du modèle...")
    results = model.evaluate(val_gen, verbose=1)
    
    print("\nMétriques d'évaluation:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
    
    return results

def predict_image(image_path, model, class_indices):
    """Fait une prédiction sur une seule image"""
    # Charger et prétraiter l'image
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=INPUT_SHAPE[:2]
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # Faire la prédiction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # Obtenir le nom de la classe prédite
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Créer le répertoire de sauvegarde du modèle si nécessaire
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Entraîner le modèle
    model, history = train()
    
    # Charger les générateurs de données pour l'évaluation
    train_gen, val_gen, class_indices = create_data_generators()
    
    # Évaluer le modèle
    evaluate_model(model, val_gen)
    
    print(f"\nModèle sauvegardé à: {MODEL_SAVE_PATH}")
    print("\nClasses disponibles:")
    for class_name, class_idx in class_indices.items():
        print(f"- {class_name} (ID: {class_idx})")
    
    # Exemple d'utilisation pour la prédiction
    # Remplacez 'chemin_vers_image.jpg' par le chemin de votre image
    # predicted_class, confidence = predict_image('chemin_vers_image.jpg', model, class_indices)
    # print(f"\nPrédiction: {predicted_class} avec une confiance de {confidence:.2f}%")
