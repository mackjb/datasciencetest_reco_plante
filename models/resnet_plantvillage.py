#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classification des maladies des plantes avec ResNet

Ce script implémente un modèle de deep learning basé sur ResNet50
pour la classification d'images de plantes et de leurs maladies.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Désactiver les avertissements inutiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.applications.resnet50 import preprocess_input

# Utilitaires
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/plantvillage/images"
OUTPUT_DIR = PROJECT_ROOT / "results/resnet_results"
MODEL_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Création des dossiers de sortie
for directory in [MODEL_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class PlantDiseaseClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=38, learning_rate=1e-4):
        """
        Initialise le classifieur de maladies des plantes avec ResNet50
        
        Args:
            input_shape: Dimensions des images d'entrée (hauteur, largeur, canaux)
            num_classes: Nombre de classes de sortie
            learning_rate: Taux d'apprentissage initial
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Construit le modèle ResNet50 avec fine-tuning"""
        # Charger ResNet50 pré-entraîné sur ImageNet
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Geler les couches du modèle de base
        base_model.trainable = False
        
        # Ajouter des couches personnalisées
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Créer le modèle
        model = Model(inputs, outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_generator, val_generator, epochs=20, batch_size=32):
        """
        Entraîne le modèle
        
        Args:
            train_generator: Générateur de données d'entraînement
            val_generator: Générateur de données de validation
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille des lots
            
        Returns:
            history: Historique d'entraînement
        """
        # Callbacks
        callbacks = [
            EarlyStopping(patience=7, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint(
                filepath=str(MODEL_DIR / 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            TensorBoard(log_dir=str(OUTPUT_DIR / 'logs'))
        ]
        
        # Entraînement
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_generator):
        """
        Évalue le modèle sur l'ensemble de test
        
        Args:
            test_generator: Générateur de données de test
            
        Returns:
            metrics: Dictionnaire des métriques d'évaluation
        """
        # Évaluation
        loss, accuracy = self.model.evaluate(test_generator, verbose=0)
        
        # Prédictions
        y_pred = self.model.predict(test_generator, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        # Métriques
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(
            y_true, y_pred_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred_classes)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }
    
    def save_model(self, path=None):
        """Sauvegarde le modèle"""
        if path is None:
            path = MODEL_DIR / 'final_model.h5'
        self.model.save(path)
        print(f"Modèle sauvegardé dans {path}")

def create_generators(data_dir, batch_size=32, img_size=(224, 224)):
    """Crée les générateurs de données pour l'entraînement, la validation et le test"""
    # Augmentation des données pour l'entraînement
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% pour la validation
    )
    
    # Pas d'augmentation pour la validation et le test
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1  # 10% pour le test
    )
    
    # Générateur d'entraînement
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Générateur de validation
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Générateur de test (utilise un sous-ensemble des données de validation)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def plot_training_history(history, output_dir):
    """Affiche et sauvegarde les courbes d'apprentissage"""
    # Courbes de perte et de précision
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Courbe de perte
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Courbe de perte')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Perte')
    ax1.legend()
    
    # Courbe de précision
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Courbe de précision')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Précision')
    ax2.legend()
    
    # Sauvegarder la figure
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    plt.close()

def plot_confusion_matrix(cm, class_names, output_dir):
    """Affiche et sauvegarde la matrice de confusion"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Entraînement d\'un modèle ResNet pour la classification des maladies des plantes')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                      help='Chemin vers le dossier contenant les images')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Taille des lots pour l\'entraînement')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Taux d\'apprentissage initial')
    args = parser.parse_args()
    
    print("Configuration de l'entraînement:")
    print(f"- Dossier de données: {args.data_dir}")
    print(f"- Taille des lots: {args.batch_size}")
    print(f"- Nombre d'époques: {args.epochs}")
    print(f"- Taux d'apprentissage: {args.learning_rate}")
    
    # Création des générateurs de données
    print("\nCréation des générateurs de données...")
    train_generator, val_generator, test_generator = create_generators(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Nombre de classes
    num_classes = len(train_generator.class_indices)
    print(f"\nNombre de classes détectées: {num_classes}")
    
    # Création du modèle
    print("\nCréation du modèle ResNet50...")
    model = PlantDiseaseClassifier(
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        learning_rate=args.learning_rate
    )
    
    # Affichage de l'architecture du modèle
    model.model.summary()
    
    # Entraînement du modèle
    print("\nDébut de l'entraînement...")
    history = model.train(
        train_generator,
        val_generator,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Évaluation du modèle
    print("\nÉvaluation sur l'ensemble de test...")
    metrics = model.evaluate(test_generator)
    
    # Affichage des résultats
    print(f"\nRésultats sur l'ensemble de test:")
    print(f"- Perte: {metrics['loss']:.4f}")
    print(f"- Précision: {metrics['accuracy']:.4f}")
    
    # Sauvegarde du modèle final
    model.save_model()
    
    # Sauvegarde des courbes d'apprentissage
    plot_training_history(history, PLOTS_DIR)
    
    # Sauvegarde de la matrice de confusion
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        metrics['class_names'],
        PLOTS_DIR
    )
    
    # Sauvegarde du rapport de classification
    with open(PLOTS_DIR / 'classification_report.json', 'w') as f:
        json.dump(metrics['report'], f, indent=4)
    
    print("\nEntraînement et évaluation terminés avec succès!")
    print(f"Résultats sauvegardés dans: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
