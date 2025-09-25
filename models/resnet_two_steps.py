import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Chemins des données et modèles ---
BASE_DIR = "/workspaces/datasciencetest_reco_plante"
DATA_DIR = os.path.join(BASE_DIR, "dataset/plantvillage/data/plantvillage_5images/segmented")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "results/models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Paramètres ---
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# --- Fonctions utilitaires ---
def get_species_from_path(path):
    """Extrait l'espèce du chemin du fichier"""
    return path.split('___')[0].split('/')[-1]

def get_disease_from_path(path):
    """Extrait la maladie du chemin du fichier"""
    return path.split('___')[-1].split('/')[0]

# --- Préparation des données ---
def prepare_datasets():
    """Prépare les datasets pour les deux modèles"""
    # Vérification du dossier de données
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Le dossier de données {DATA_DIR} n'existe pas.")
    
    # Récupération des chemins et des labels
    image_paths = []
    species_labels = []
    disease_labels = []
    
    for root, _, files in os.walk(DATA_DIR):
        if not files:
            continue
            
        # Extraction des labels depuis le chemin
        rel_path = os.path.relpath(root, DATA_DIR)
        if '__' in rel_path:  # Format: Espace___Maladie
            species = rel_path.split('___')[0]
            disease = rel_path.split('___')[1]
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
                    species_labels.append(species)
                    disease_labels.append(f"{species}___{disease}")
    
    # Création des DataFrames
    df = pd.DataFrame({
        'image_path': image_paths,
        'species': species_labels,
        'disease': disease_labels
    })
    
    # Encodage des labels
    species_encoder = {v: i for i, v in enumerate(df['species'].unique())}
    disease_encoder = {v: i for i, v in enumerate(df['disease'].unique())}
    
    df['species_label'] = df['species'].map(species_encoder)
    df['disease_label'] = df['disease'].map(disease_encoder)
    
    # Séparation train/validation
    train_df = df.sample(frac=1-VALIDATION_SPLIT, random_state=42)
    val_df = df.drop(train_df.index)
    
    return train_df, val_df, species_encoder, disease_encoder

# --- Générateurs de données ---
class DataGenerator(tf.keras.utils.Sequence):
    """Générateur personnalisé pour charger les images et les labels"""
    def __init__(self, df, batch_size=32, target_size=(224, 224), mode='species'):
        self.df = df
        self.batch_size = batch_size
        self.target_size = target_size
        self.mode = mode  # 'species' ou 'disease'
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        # Chargement des images
        images = []
        for path in batch_df['image_path']:
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            images.append(img)
        
        images = np.array(images)
        
        # Sélection des labels en fonction du mode
        if self.mode == 'species':
            labels = tf.keras.utils.to_categorical(batch_df['species_label'], num_classes=len(species_encoder))
        else:  # disease
            labels = tf.keras.utils.to_categorical(batch_df['disease_label'], num_classes=len(disease_encoder))
        
        return images, labels
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- Construction des modèles ---
def build_species_model(num_classes):
    """Modèle pour la détection d'espèces"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    
    inputs = Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_disease_model(num_classes):
    """Modèle pour la détection de maladies"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    
    inputs = Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),  # Taux d'apprentissage plus bas
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Callbacks ---
def get_callbacks(model_name):
    """Retourne les callbacks pour l'entraînement"""
    return [
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, f"best_{model_name}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(BASE_DIR, 'results/logs', model_name),
            histogram_freq=1
        )
    ]

# --- Entraînement ---
def train_models():
    """Entraîne les deux modèles"""
    # Préparation des données
    train_df, val_df, species_enc, disease_enc = prepare_datasets()
    
    # Création des générateurs
    train_species_gen = DataGenerator(train_df, BATCH_SIZE, INPUT_SHAPE[:2], 'species')
    val_species_gen = DataGenerator(val_df, BATCH_SIZE, INPUT_SHAPE[:2], 'species')
    
    train_disease_gen = DataGenerator(train_df, BATCH_SIZE, INPUT_SHAPE[:2], 'disease')
    val_disease_gen = DataGenerator(val_df, BATCH_SIZE, INPUT_SHAPE[:2], 'disease')
    
    # Entraînement du modèle espèces
    print("\n=== Entraînement du modèle d'espèces ===")
    species_model = build_species_model(len(species_enc))
    species_history = species_model.fit(
        train_species_gen,
        validation_data=val_species_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks('species_model'),
        verbose=1
    )
    
    # Entraînement du modèle maladies
    print("\n=== Entraînement du modèle de maladies ===")
    disease_model = build_disease_model(len(disease_enc))
    disease_history = disease_model.fit(
        train_disease_gen,
        validation_data=val_disease_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks('disease_model'),
        verbose=1
    )
    
    return species_model, disease_model, species_enc, disease_enc, species_history, disease_history

# --- Évaluation ---
def evaluate_models(species_model, disease_model, val_df, species_enc, disease_enc):
    """Évalue les modèles et génère des rapports"""
    # Création des générateurs d'évaluation
    val_species_gen = DataGenerator(val_df, BATCH_SIZE, INPUT_SHAPE[:2], 'species')
    val_disease_gen = DataGenerator(val_df, BATCH_SIZE, INPUT_SHAPE[:2], 'disease')
    
    # Évaluation espèces
    print("\n=== Évaluation du modèle d'espèces ===")
    species_results = species_model.evaluate(val_species_gen)
    print(f"Perte: {species_results[0]:.4f}, Précision: {species_results[1]:.4f}")
    
    # Évaluation maladies
    print("\n=== Évaluation du modèle de maladies ===")
    disease_results = disease_model.evaluate(val_disease_gen)
    print(f"Perte: {disease_results[0]:.4f}, Précision: {disease_results[1]:.4f}")
    
    # Prédictions pour le rapport de classification
    val_species_gen = DataGenerator(val_df, BATCH_SIZE, INPUT_SHAPE[:2], 'species')
    val_disease_gen = DataGenerator(val_df, BATCH_SIZE, INPUT_SHAPE[:2], 'disease')
    
    # Prédictions espèces
    y_true_species = []
    y_pred_species = []
    for i in range(len(val_species_gen)):
        x, y = val_species_gen[i]
        y_true_species.extend(np.argmax(y, axis=1))
        y_pred_species.extend(np.argmax(species_model.predict(x), axis=1))
    
    # Prédictions maladies
    y_true_disease = []
    y_pred_disease = []
    for i in range(len(val_disease_gen)):
        x, y = val_disease_gen[i]
        y_true_disease.extend(np.argmax(y, axis=1))
        y_pred_disease.extend(np.argmax(disease_model.predict(x), axis=1))
    
    # Création des rapports
    species_report = classification_report(
        y_true_species,
        y_pred_species,
        target_names=list(species_enc.keys()),
        output_dict=True
    )
    
    disease_report = classification_report(
        y_true_disease,
        y_pred_disease,
        target_names=list(disease_enc.keys()),
        output_dict=True
    )
    
    # Conversion en DataFrame pour une meilleure lisibilité
    species_report_df = pd.DataFrame(species_report).transpose()
    disease_report_df = pd.DataFrame(disease_report).transpose()
    
    # Sauvegarde des rapports
    os.makedirs(os.path.join(BASE_DIR, 'results/reports'), exist_ok=True)
    species_report_df.to_csv(os.path.join(BASE_DIR, 'results/reports/species_classification_report.csv'))
    disease_report_df.to_csv(os.path.join(BASE_DIR, 'results/reports/disease_classification_report.csv'))
    
    return species_report_df, disease_report_df

# --- Fonction principale ---
if __name__ == "__main__":
    print("Démarrage de l'entraînement en deux étapes...")
    
    # Entraînement des modèles
    species_model, disease_model, species_enc, disease_enc, species_hist, disease_hist = train_models()
    
    # Préparation des données pour l'évaluation
    train_df, val_df, _, _ = prepare_datasets()
    
    # Évaluation et génération des rapports
    species_report, disease_report = evaluate_models(
        species_model, disease_model, val_df, species_enc, disease_enc
    )
    
    print("\n=== Rapports de classification sauvegardés dans results/reports/ ===")
    print("1. species_classification_report.csv")
    print("2. disease_classification_report.csv")
    
    # Affichage des précisions moyennes
    print("\n=== Précision moyenne par modèle ===")
    print(f"Modèle espèces: {species_report.loc['accuracy', 'precision']*100:.2f}%")
    print(f"Modèle maladies: {disease_report.loc['accuracy', 'precision']*100:.2f}%")
    
    print("\nEntraînement et évaluation terminés !")
