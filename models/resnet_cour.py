import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)
import matplotlib.pyplot as plt
from datetime import datetime
import json

# --- Chemins des données et modèles ---
BASE_DIR = "/workspaces/datasciencetest_reco_plante"
DATA_DIR = os.path.join(BASE_DIR, "dataset/plantvillage/data/plantvillage dataset/segmented")
MODELS_DIR = os.path.join(BASE_DIR, "results/models")
LOGS_DIR = os.path.join(BASE_DIR, "results/logs")

# Création des dossiers si inexistants
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Paramètres optimisés pour CPU ---
INPUT_SHAPE = (160, 160, 3)  # Taille d'image réduite
BATCH_SIZE = 16  # Batch plus petit pour éviter les problèmes de mémoire
INITIAL_EPOCHS = 10  # Moins d'époques pour la première phase
FINE_TUNE_EPOCHS = 5  # Moins d'époques pour le fine-tuning
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
VALIDATION_SPLIT = 0.2
CLASS_MODE = 'categorical'
AUGMENTATION = True

# Configuration du GPU pour une meilleure performance
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configuré avec succès")
    except RuntimeError as e:
        print(f"Erreur de configuration GPU: {e}")
else:
    print("Aucun GPU détecté, utilisation du CPU")

# --- Générateurs de données avec augmentation ---
def create_data_generators():
    """Crée les générateurs d'images pour l'entraînement et la validation"""
    # Vérifier si le dossier de données existe
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Le dossier de données {DATA_DIR} n'existe pas.")
    
    # Liste des classes (sous-dossiers)
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not classes:
        raise ValueError(f"Aucune classe trouvée dans {DATA_DIR}")
    
    print(f"\n{'='*50}")
    print(f"Chargement des données depuis: {DATA_DIR}")
    print(f"{len(classes)} classes détectées")
    print("="*50)
    
    # Augmentation de données pour l'entraînement
    if AUGMENTATION:
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=VALIDATION_SPLIT
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            validation_split=VALIDATION_SPLIT
        )

    # Pas d'augmentation pour la validation
    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=VALIDATION_SPLIT
    )
    
    # Générateur d'entraînement
    print("\nCréation des générateurs de données...")
    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset='training',
        shuffle=True,
        seed=42
    )

    # Générateur de validation
    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Afficher les statistiques des données
    print("\nStatistiques des données:")
    print(f"- Nombre total d'images: {train_gen.samples + val_gen.samples}")
    print(f"- Images d'entraînement: {train_gen.samples}")
    print(f"- Images de validation: {val_gen.samples}")
    print(f"- Taille du batch: {BATCH_SIZE}")
    print(f"- Étapes par époque (train): {len(train_gen)}")
    print(f"- Étapes par époque (val): {len(val_gen)}")
    
    # Sauvegarder le mapping des classes
    class_indices = train_gen.class_indices
    class_mapping = {v: k for k, v in class_indices.items()}
    class_mapping_path = os.path.join(MODELS_DIR, 'class_mapping.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"\nMapping des classes sauvegardé dans: {class_mapping_path}")
    
    return train_gen, val_gen, class_indices

def build_model(num_classes):
    """Construit le modèle ResNet50 avec une nouvelle tête"""
    print("\nConstruction du modèle ResNet50...")
    
    # Charger ResNet50 pré-entraîné sur ImageNet
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE,
        pooling=None
    )
    
    # Geler les couches de base dans un premier temps
    base_model.trainable = False
    
    # Construire le modèle
    inputs = Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Ajout de couches denses avec normalisation par lots
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Dropout(0.3)(x)
    
    # Couche de sortie
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Créer le modèle
    model = Model(inputs, outputs)
    
    # Afficher un résumé du modèle
    model.summary()
    
    return model, base_model

def train():
    """Entraîne le modèle avec une approche en deux étapes"""
    # Créer les générateurs de données
    train_gen, val_gen, class_indices = create_data_generators()
    num_classes = len(class_indices)
    
    # Construire le modèle
    model, base_model = build_model(num_classes)
    
    # Étape 1: Entraînement de la tête uniquement
    print("\n" + "="*50)
    print("ÉTAPE 1: Entraînement de la tête du modèle")
    print("="*50)
    
    # Compiler le modèle avec un taux d'apprentissage plus élevé
    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Chemins de sauvegarde
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'resnet50_plant_disease.h5')
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
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
        ),
        CSVLogger(os.path.join(log_dir, 'training_log.csv')),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='batch'
        )
    ]
    
    # Entraînement initial (tête uniquement)
    print("\nDémarrage de l'entraînement de la tête...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Étape 2: Fine-tuning des dernières couches de ResNet
    print("\n" + "="*50)
    print("ÉTAPE 2: Fine-tuning du modèle complet")
    print("="*50)
    
    # Débloquer les dernières couches de ResNet
    base_model.trainable = True
    
    # Geler les premières couches (les premières apprennent des motifs plus génériques)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompiler avec un taux d'apprentissage plus faible
    optimizer = Adam(learning_rate=FINE_TUNE_LR)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Réinitialiser les callbacks pour le fine-tuning
    fine_tune_callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH.replace('.h5', '_finetuned.h5'),
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
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Fine-tuning
    print("\nDémarrage du fine-tuning...")
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=history.epoch[-1],
        epochs=history.epoch[-1] + FINE_TUNE_EPOCHS,
        callbacks=fine_tune_callbacks,
        verbose=1
    )
    
    # Combiner les historiques
    combined_history = {}
    for key in history.history.keys():
        combined_history[key] = history.history[key] + history_fine.history[key]
    
    return model, combined_history

def evaluate_model(model, val_gen, class_indices):
    """Évalue le modèle sur l'ensemble de validation et génère des visualisations"""
    print("\n" + "="*50)
    print("ÉVALUATION DU MODÈLE")
    print("="*50)
    
    # Évaluation complète
    results = model.evaluate(val_gen, verbose=1, return_dict=True)
    
    # Affichage des métriques
    print("\n=== MÉTRIQUES D'ÉVALUATION ===")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    
    # Prédictions pour la matrice de confusion
    print("\nGénération des prédictions...")
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Matrice de confusion
    plot_confusion_matrix(y_true, y_pred, class_indices)
    
    # Courbes d'apprentissage
    plot_training_history(history)
    
    return results

def predict_image(image_path, model, class_indices, top_k=3):
    """
    Fait une prédiction sur une seule image
    
    Args:
        image_path: Chemin vers l'image à prédire
        model: Modèle chargé
        class_indices: Dictionnaire de mapping des classes
        top_k: Nombre de meilleures prédictions à retourner
        
    Returns:
        tuple: (classe prédite, confiance, top_k_predictions)
    """
    # Charger et prétraiter l'image
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=INPUT_SHAPE[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        # Faire la prédiction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Obtenir les indices des top_k prédictions
        top_k_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Créer une liste des prédictions avec leurs probabilités
        top_k_predictions = [
            (class_indices[str(i)], float(predictions[i])) 
            for i in top_k_indices
        ]
        
        # Classe prédite et confiance
        predicted_class = top_k_predictions[0][0]
        confidence = top_k_predictions[0][1] * 100
        
        return predicted_class, confidence, top_k_predictions
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        return None, 0.0, []

def plot_confusion_matrix(y_true, y_pred, class_indices):
    """Affiche la matrice de confusion"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Créer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Tracer avec seaborn
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_indices.values(),
                yticklabels=class_indices.values())
    plt.title('Matrice de Confusion')
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédictions')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Sauvegarder la figure
    confusion_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"\nMatrice de confusion sauvegardée: {confusion_path}")
    plt.close()

def plot_training_history(history):
    """Affiche les courbes d'apprentissage"""
    # Préparer les données
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    # Créer les figures
    plt.figure(figsize=(16, 6))
    
    # Tracer la précision
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Précision entraînement')
    plt.plot(epochs_range, val_acc, label='Précision validation')
    plt.legend(loc='lower right')
    plt.title('Précision du modèle')
    
    # Tracer la perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perte entraînement')
    plt.plot(epochs_range, val_loss, label='Perte validation')
    plt.legend(loc='upper right')
    plt.title('Perte du modèle')
    
    # Sauvegarder la figure
    training_plot_path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
    print(f"Courbes d'apprentissage sauvegardées: {training_plot_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Entraînement et évaluation du modèle de détection de maladies des plantes')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'predict'],
                       help='Mode d\'exécution: train, evaluate, ou predict')
    parser.add_argument('--image', type=str, 
                       help='Chemin vers l\'image pour la prédiction (mode predict uniquement)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Entraîner le modèle
        print("Démarrage de l'entraînement...")
        model, history = train()
        
        # Charger les générateurs de données pour l'évaluation
        train_gen, val_gen, class_indices = create_data_generators()
        
        # Évaluer le modèle
        evaluate_model(model, val_gen, class_indices)
        
        # Afficher les informations finales
        print(f"\n{'='*50}")
        print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        print(f"{'='*50}")
        print(f"Modèle sauvegardé à: {MODEL_SAVE_PATH}")
        print(f"Logs TensorBoard: {os.path.join(LOGS_DIR, 'training_*')}")
        print(f"Résultats dans: {RESULTS_DIR}")
        
    elif args.mode == 'evaluate':
        # Charger le modèle et les données pour l'évaluation
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Modèle non trouvé: {MODEL_SAVE_PATH}")
            
        # Charger le modèle
        model = load_model(MODEL_SAVE_PATH)
        
        # Charger les générateurs de données
        train_gen, val_gen, class_indices = create_data_generators()
        
        # Évaluer le modèle
        evaluate_model(model, val_gen, class_indices)
        
    elif args.mode == 'predict':
        if not args.image:
            raise ValueError("Veuillez spécifier le chemin vers une image avec --image")
            
        # Charger le modèle
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Modèle non trouvé: {MODEL_SAVE_PATH}")
            
        model = load_model(MODEL_SAVE_PATH)
        
        # Charger le mapping des classes
        class_mapping_path = os.path.join(MODELS_DIR, 'class_mapping.json')
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        # Faire la prédiction
        predicted_class, confidence, top_predictions = predict_image(args.image, model, class_mapping)
        
        # Afficher les résultats
        print("\n" + "="*50)
        print("RÉSULTATS DE LA PRÉDICTION")
        print("="*50)
        print(f"Image: {args.image}")
        print(f"\nTop {len(top_predictions)} prédictions:")
        for i, (class_name, prob) in enumerate(top_predictions, 1):
            print(f"{i}. {class_name}: {prob*100:.2f}%")
        
        print(f"\nMeilleure prédiction: {predicted_class} (Confiance: {confidence:.2f}%)")
    
    else:
        parser.print_help()
