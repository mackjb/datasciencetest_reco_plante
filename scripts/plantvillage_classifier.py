#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classification d'espèces et de maladies de plantes avec Deep Learning

Ce script implémente un modèle de deep learning pour la classification d'espèces
de plantes et de leurs maladies en utilisant le dataset PlantVillage.
"""

# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------
# Bibliothèques standard
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Traitement des données
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Deep Learning
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)

# Vision par ordinateur
import cv2

# Interprétabilité
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# Désactiver les avertissements inutiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
class Config:
    """Configuration du modèle et des chemins
    
    Cette classe gère tous les paramètres de configuration pour l'entraînement et l'évaluation du modèle.
    
    Attributs:
        img_size (tuple): Dimensions des images d'entrée (hauteur, largeur)
        batch_size (int): Taille des lots pour l'entraînement
        epochs (int): Nombre d'époques d'entraînement
        learning_rate (float): Taux d'apprentissage initial
        dropout_rate (float): Taux de dropout pour la couche de classification
        fine_tune_layers (int): Nombre de couches à dégeler pour le fine-tuning
        data_path (Path): Chemin vers le dossier des données
        image_dir (Path): Chemin vers le dossier des images
        label_csv (Path): Chemin vers le fichier CSV des labels
        output_dir (Path): Dossier de sortie principal
        model_dir (Path): Dossier de sauvegarde des modèles
        logs_dir (Path): Dossier des logs TensorBoard
        plots_dir (Path): Dossier de sauvegarde des graphiques
        model_path (Path): Chemin de sauvegarde du meilleur modèle
        metrics_file (Path): Fichier de sauvegarde des métriques
        augmentation (dict): Paramètres d'augmentation des données
        class_names (list): Liste des noms de classes
        random_state (int): Graine pour la reproductibilité
    """
    def __init__(self):
        # Paramètres du modèle
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 20
        self.learning_rate = 1e-3
        self.dropout_rate = 0.3
        self.fine_tune_layers = 50  # Nombre de couches à dégeler pour le fine-tuning
        self.num_classes = None  # Sera défini lors du chargement des données
        
        # Chemins des données
        self.data_path = Path("dataset/plantvillage/")
        self.image_dir = self.data_path / "images"
        self.label_csv = self.data_path / "labels.csv"
        
        # Dossiers de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"results/plantvillage_{timestamp}")
        self.model_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        
        # Création des dossiers si nécessaire
        for directory in [self.model_dir, self.logs_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Fichiers de sortie
        self.model_path = self.model_dir / "best_model.h5"
        self.metrics_file = self.output_dir / "metrics.json"
        
        # Configuration de l'augmentation des données
        self.augmentation = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': [0.8, 1.2],
            'zoom_range': 0.2,
            'fill_mode': 'nearest'
        }
        
        # Noms des classes (seront mis à jour lors du chargement des données)
        self.class_names = []
        
        # Paramètres de reproductibilité
        self.random_state = 42
        
    def save(self, filepath=None):
        """Sauvegarde la configuration dans un fichier JSON
        
        Args:
            filepath (str, optional): Chemin du fichier de sortie. Si None, utilise self.output_dir/config.json
            
        Returns:
            Path: Chemin du fichier de configuration sauvegardé
        """
        if filepath is None:
            filepath = self.output_dir / "config.json"
            
        config_dict = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'fine_tune_layers': self.fine_tune_layers,
            'data_path': str(self.data_path),
            'output_dir': str(self.output_dir),
            'model_path': str(self.model_path),
            'logs_dir': str(self.logs_dir),
            'plots_dir': str(self.plots_dir),
            'augmentation': self.augmentation,
            'class_names': self.class_names,
            'random_state': self.random_state
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Charge une configuration depuis un fichier JSON
        
        Args:
            filepath (str): Chemin vers le fichier de configuration
            
        Returns:
            Config: Instance de la classe Config chargée
            
        Raises:
            FileNotFoundError: Si le fichier de configuration n'existe pas
            json.JSONDecodeError: Si le fichier n'est pas un JSON valide
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                # Conversion des chemins en objets Path
                if key.endswith('_path') or key.endswith('_dir') or key == 'data_path':
                    setattr(config, key, Path(value))
                else:
                    setattr(config, key, value)
        
        return config

# ------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ------------------------------------------------------
class PlantVillageDataLoader:
    """Classe pour charger et prétraiter les données PlantVillage de manière efficace
    
    Cette classe utilise ImageDataGenerator.flow_from_directory pour charger les images
    à la volée, ce qui est plus efficace en mémoire pour les grands jeux de données.
    """
    
    def __init__(self, config):
        self.config = config
        self.class_names = []
        
    def _get_class_names(self, directory):
        """Récupère la liste des noms de classes depuis la structure des dossiers"""
        return sorted([d for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d))])
    
    def load_data(self):
        """Charge les données en utilisant des générateurs pour économiser la mémoire
        
        Returns:
            tuple: (train_generator, val_generator, test_generator, class_names)
        """
        print("\nChargement des données...")
        
        # Vérification de la structure des dossiers
        if not self.config.image_dir.exists():
            raise FileNotFoundError(f"Dossier d'images non trouvé: {self.config.image_dir}")
            
        # Récupération des noms de classes depuis la structure des dossiers
        self.class_names = self._get_class_names(self.config.image_dir)
        self.config.num_classes = len(self.class_names)
        print(f"{self.config.num_classes} classes détectées: {', '.join(self.class_names[:5])}...")
        
        # Création des générateurs de données
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,  # 20% pour la validation
            **self.config.augmentation
        )
        
        # Générateur d'entraînement
        train_generator = train_datagen.flow_from_directory(
            self.config.image_dir,
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='training',
            seed=self.config.random_state
        )
        
        # Générateur de validation
        val_generator = train_datagen.flow_from_directory(
            self.config.image_dir,
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='validation',
            seed=self.config.random_state
        )
        
        # Générateur de test (sans augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Créer un dossier temporaire pour les données de test
        test_dir = Path(self.config.output_dir) / "test_split"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer un générateur pour le test (20% des données)
        test_generator = test_datagen.flow_from_directory(
            self.config.image_dir,
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.random_state
        )
        
        # Calculer les tailles des jeux de données
        total_samples = len(test_generator.filenames)
        test_size = int(0.2 * total_samples)
        train_size = total_samples - test_size
        val_size = int(0.2 * train_size)
        train_size -= val_size
        
        print(f"Données chargées : {total_samples} images, {self.config.num_classes} classes")
        print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")
        
        return train_generator, val_generator, test_generator, self.class_names
# ------------------------------------------------------
# MODÈLE
# ------------------------------------------------------
class PlantDiseaseClassifier:
    """Classe pour le modèle de classification des maladies de plantes
    
    Cette classe gère la création, l'entraînement et l'évaluation d'un modèle
    de classification basé sur ResNet50 avec fine-tuning.
    """
    
    def __init__(self, config):
        """Initialise le classifieur avec la configuration fournie
        
        Args:
            config: Objet de configuration contenant les paramètres du modèle
        """
        self.config = config
        self.num_classes = config.num_classes
        self.model = None
        self.base_model = None  # Référence au modèle de base pour un accès facile
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialise le modèle en créant un nouveau ou en chargeant un existant"""
        if hasattr(self.config, 'model_path') and self.config.model_path.exists():
            self._load_existing_model()
        else:
            self._build_new_model()
    
    def _build_new_model(self):
        """Construit un nouveau modèle à partir de zéro"""
        print("Construction d'un nouveau modèle...")
        self.model = self._build_model()
        
    def _load_existing_model(self):
        """Charge un modèle existant depuis le disque"""
        print(f"Chargement du modèle depuis {self.config.model_path}")
        try:
            # Charger le modèle complet
            self.model = load_model(self.config.model_path)
            
            # Récupérer la référence au modèle de base
            # On suppose que le modèle a une architecture connue avec le modèle de base en première couche
            if len(self.model.layers) > 1 and isinstance(self.model.layers[1], Model):
                self.base_model = self.model.layers[1]
                print("Modèle de base chargé avec succès")
            else:
                print("Avertissement: Impossible de trouver le modèle de base dans le modèle chargé")
                
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            print("Construction d'un nouveau modèle...")
            self._build_new_model()
    
    def _build_model(self):
        """Construit le modèle ResNet50 avec fine-tuning
        
        Returns:
            tf.keras.Model: Modèle compilé avec ResNet50 comme extracteur de caractéristiques
        """
        print("\nConstruction du modèle...")
        
        # Chargement du modèle de base (sans la couche de classification)
        self.base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.img_size + (3,),
            pooling=None
        )
        
        # Geler toutes les couches du modèle de base initialement
        self.base_model.trainable = False
        
        # Construction du modèle complet
        inputs = tf.keras.Input(shape=self.config.img_size + (3,))
        
        # Normalisation des entrées
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        
        # Passage dans le modèle de base
        x = self.base_model(x, training=False)
        
        # Couches de classification
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Création du modèle
        model = Model(inputs, outputs, name='plant_disease_classifier')
        
        # Compilation du modèle
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            run_eagerly=False
        )
        
        # Affichage du résumé
        model.summary()
        return model
    
    def train(self, train_generator, val_generator):
        """Entraîne le modèle avec les générateurs de données
        
        Args:
            train_generator: Générateur de données d'entraînement (avec augmentation)
            val_generator: Générateur de données de validation
            
        Returns:
            History: Historique d'entraînement
        """
        # Création des dossiers de sortie
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callbacks = [
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
            ModelCheckpoint(
                filepath=str(self.config.model_dir / 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            TensorBoard(
                log_dir=str(self.config.logs_dir),
                histogram_freq=1
            )
        ]
        
        # Calcul des étapes par époque
        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = val_generator.samples // val_generator.batch_size
        
        # Phase 1: Entraînement initial (couches gelées)
        print("\n=== PHASE 1: Entraînement initial (couches gelées) ===")
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.epochs // 2,  # Moitié des époques pour la phase 1
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=True,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (dégel des dernières couches)
        print("\n=== PHASE 2: Fine-tuning des dernières couches ===")
        self._unfreeze_layers()
        
        # Réduction du taux d'apprentissage pour le fine-tuning
        reduced_lr = self.config.learning_rate / 10
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, reduced_lr)
        
        # Réentraînement avec les dernières couches dégelées
        history_fine = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.epochs,
            initial_epoch=history.epoch[-1] + 1,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=True,
            verbose=1
        )
        
        # Fusion des historiques
        for k in history.history:
            if k in history_fine.history:
                history.history[k].extend(history_fine.history[k])
        
        # Sauvegarde du modèle final
        self.model.save(self.config.model_dir / 'final_model.h5')
        
        return history
    
    def _unfreeze_layers(self):
        """Dégèle les dernières couches du modèle de base"""
        print("\nDégel des dernières couches pour le fine-tuning...")
        
        # Rendre le modèle de base entraînable
        self.model.layers[1].trainable = True
        
        # Geler les premières couches et ne garder que les dernières dégelées
        for layer in self.model.layers[1].layers[:-self.config.fine_tune_layers]:
            layer.trainable = False
        
        # Recompilation du modèle pour appliquer les changements
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics_names
        )
        
        # Affichage du résumé du modèle
        self.model.summary()
    
    def evaluate(self, test_generator):
        """Évalue le modèle sur l'ensemble de test
        
        Args:
            test_generator: Générateur de données de test
            
        Returns:
            dict: Dictionnaire contenant les métriques d'évaluation
        """
        print("\nÉvaluation sur l'ensemble de test...")
        
        # Réinitialisation du générateur
        test_generator.reset()
        
        # Évaluation du modèle
        start_time = time.time()
        test_loss, test_accuracy, test_auc = self.model.evaluate(
            test_generator,
            steps=test_generator.samples // test_generator.batch_size,
            verbose=1
        )
        inference_time = time.time() - start_time
        
        # Prédictions pour les métriques supplémentaires
        y_true = []
        y_pred_probs = []
        
        # Parcourir le générateur pour récupérer les vraies étiquettes et les prédictions
        for i in range(int(np.ceil(test_generator.samples / test_generator.batch_size))):
            batch_x, batch_y = next(test_generator)
            batch_pred = self.model.predict(batch_x, verbose=0)
            y_true.extend(np.argmax(batch_y, axis=1))
            y_pred_probs.extend(batch_pred)
        
        # Conversion en tableaux numpy
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Métriques
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Rapport de classification détaillé
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.config.class_names,
            output_dict=True
        )
        
        # Affichage des résultats
        print(f"\nRésultats sur l'ensemble de test:")
        print(f"- Exactitude (Accuracy): {accuracy:.4f}")
        print(f"- F1 Macro: {f1_macro:.4f}")
        print(f"- F1 Pondéré: {f1_weighted:.4f}")
        print(f"- Temps d'inférence: {inference_time:.2f} secondes")
        print(f"- Temps moyen par image: {(inference_time/len(X_test)*1000):.2f} ms")
        
        # Sauvegarde des métriques
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'inference_time': inference_time,
            'avg_inference_time': inference_time/len(X_test),
            'classification_report': class_report
        }
        
        with open(self.config.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Matrice de confusion
        self._plot_confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Affiche et sauvegarde la matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.config.class_names,
            yticklabels=self.config.class_names
        )
        
        plt.title('Matrice de confusion')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies classes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarde de la figure
        cm_path = self.config.plots_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Matrice de confusion sauvegardée dans {cm_path}")
    
    def plot_training_history(self, history):
        """Affiche et sauvegarde les courbes d'apprentissage"""
        # Précision
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Précision du modèle')
        plt.ylabel('Précision')
        plt.xlabel('Époque')
        plt.legend()
        
        # Perte
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Perte du modèle')
        plt.ylabel('Perte')
        plt.xlabel('Époque')
        plt.legend()
        
        # Sauvegarde de la figure
        plt.tight_layout()
        history_path = self.config.plots_dir / 'training_history.png'
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Courbes d'apprentissage sauvegardées dans {history_path}")
    
    def explain_predictions(self, X_sample, class_names, num_samples=3):
        """Génère des explications pour les prédictions"""
        print("\nGénération des explications...")
        
        # Sélection d'un échantillon
        if len(X_sample) > num_samples:
            indices = np.random.choice(len(X_sample), num_samples, replace=False)
            X_sample = X_sample[indices]
        
        # Explications avec Grad-CAM
        self._explain_with_gradcam(X_sample, class_names)
        
        # Explications avec LIME
        self._explain_with_lime(X_sample, class_names)
        
        # Explications avec SHAP (optionnel, peut être long)
        if len(X_sample) <= 5:  # Limiter le nombre d'échantillons pour SHAP
            self._explain_with_shap(X_sample, class_names)
    
    def _explain_with_gradcam(self, X_sample, class_names):
        """Génère des explications avec Grad-CAM"""
        print("  - Génération des explications Grad-CAM...")
        
        # Couche intermédiaire pour Grad-CAM
        last_conv_layer_name = 'conv5_block3_out'
        
        # Création du modèle pour Grad-CAM
        grad_model = Model(
            [self.model.inputs],
            [self.model.get_layer(f'resnet50').get_layer(last_conv_layer_name).output, 
             self.model.output]
        )
        
        for i, img in enumerate(X_sample):
            # Prédiction
            img_array = np.expand_dims(img, axis=0)
            preds = self.model.predict(img_array)
            pred_class = np.argmax(preds[0])
            
            # Calcul de la heatmap
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, pred_class]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Affichage
            plt.figure(figsize=(10, 5))
            
            # Image originale
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Original - {class_names[pred_class]} ({preds[0][pred_class]:.2f})")
            plt.axis('off')
            
            # Heatmap
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.imshow(
                tf.keras.preprocessing.image.array_to_img(
                    tf.expand_dims(heatmap, -1) * 255, 
                    scale=False
                ),
                alpha=0.4,
                cmap='jet'
            )
            plt.title("Grad-CAM")
            plt.axis('off')
            
            # Sauvegarde
            gradcam_path = self.config.plots_dir / f'gradcam_explanation_{i+1}.png'
            plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"    Explications Grad-CAM sauvegardées dans {self.config.plots_dir}/gradcam_*.png")
    
    def _explain_with_lime(self, X_sample, class_names, num_samples=1000):
        """Génère des explications avec LIME"""
        print("  - Génération des explications LIME...")
        
        # Fonction de prédiction compatible avec LIME
        def predict_fn(images):
            return self.model.predict(images)
        
        # Création de l'explainer LIME
        explainer = lime_image.LimeImageExplainer()
        
        for i, img in enumerate(X_sample):
            # Explication
            explanation = explainer.explain_instance(
                img,
                classifier_fn=predict_fn,
                top_labels=3,
                hide_color=0,
                num_samples=num_samples
            )
            
            # Affichage de l'explication
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            plt.figure(figsize=(10, 5))
            
            # Image originale
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Image originale")
            plt.axis('off')
            
            # Explication LIME
            plt.subplot(1, 2, 2)
            plt.imshow(mark_boundaries(temp / 255.0, mask))
            plt.title(f"Explication LIME - {class_names[explanation.top_labels[0]]}")
            plt.axis('off')
            
            # Sauvegarde
            lime_path = self.config.plots_dir / f'lime_explanation_{i+1}.png'
            plt.savefig(lime_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"    Explications LIME sauvegardées dans {self.config.plots_dir}/lime_*.png")
    
    def _explain_with_shap(self, X_sample, class_names):
        """Génère des explications avec SHAP (peut être long)"""
        print("  - Génération des explications SHAP (peut prendre du temps)...")
        
        # Création de l'explainer SHAP
        background = X_sample[np.random.choice(X_sample.shape[0], 10, replace=False)]
        explainer = shap.DeepExplainer(self.model, background)
        
        # Calcul des valeurs SHAP
        shap_values = explainer.shap_values(X_sample)
        
        # Affichage des explications
        plt.figure(figsize=(10, 10))
        shap.image_plot(
            shap_values,
            X_sample,
            labels=[class_names[i] for i in range(len(class_names))],
            show=False
        )
        
        # Sauvegarde
        shap_path = self.config.plots_dir / 'shap_explanations.png'
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Explications SHAP sauvegardées dans {shap_path}")
    
    def save_model(self, model_name='best_model'):
        """Sauvegarde le modèle et la configuration
        
        Args:
            model_name (str): Nom du modèle à sauvegarder (sans extension)
        """
        # Création du dossier de sortie s'il n'existe pas
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemin de sauvegarde du modèle
        model_path = self.config.model_dir / f"{model_name}.h5"
        
        # Sauvegarde du modèle
        self.model.save(model_path)
        print(f"\nModèle sauvegardé dans {model_path}")
        
        # Sauvegarde de la configuration
        config_path = self.config.output_dir / 'config.json'
        self.config.save(config_path)
        print(f"Configuration sauvegardée dans {config_path}")
    
    @classmethod
    def load_model(cls, config_path):
        """Charge un modèle existant à partir d'un fichier de configuration
        
        Args:
            config_path (str or Path): Chemin vers le fichier de configuration
            
        Returns:
            PlantDiseaseClassifier: Instance du modèle chargé
        """
        # Chargement de la configuration
        config = Config.load(config_path)
        
        # Création de l'instance
        classifier = cls(config)
        
        # Chargement des poids du modèle
        model_path = config.model_dir / 'best_model.h5'
        if not model_path.exists():
            model_path = config.model_dir / 'final_model.h5'
            
        if model_path.exists():
            print(f"Chargement des poids du modèle depuis {model_path}")
            classifier.model = load_model(model_path)
            
            # Mise à jour de la référence au modèle de base
            if len(classifier.model.layers) > 1 and isinstance(classifier.model.layers[1], Model):
                classifier.base_model = classifier.model.layers[1]
                print("Modèle de base chargé avec succès")
        else:
            print(f"Avertissement: Aucun modèle trouvé dans {config.model_dir}")
            print("Initialisation d'un nouveau modèle...")
        
        return classifier

# FONCTION PRINCIPALE
# ------------------------------------------------------
def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Classification des maladies des plantes avec Deep Learning')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'evaluate', 'explain'],
                      help='Mode d\'exécution: train, evaluate ou explain')
    parser.add_argument('--model-dir', type=str, help='Chemin vers le modèle à charger')
    parser.add_argument('--no-augmentation', action='store_true',
                      help='Désactiver l\'augmentation des données')
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.mode = args.mode
    
    # Création des dossiers de sortie
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Désactivation de l'augmentation si nécessaire
    if args.no_augmentation:
        config.augmentation = {}
    
    # Sauvegarde de la configuration
    config.save(config.output_dir / 'config.json')
    
    # Chargement des données
    print("Chargement des données...")
    data_loader = PlantVillageDataLoader(config)
    train_generator, val_generator, test_generator, class_names = data_loader.load_data()
    
    # Mise à jour des noms de classes dans la configuration
    config.class_names = class_names
    
    # Initialisation ou chargement du modèle
    if args.model_dir:
        print(f"\nChargement du modèle depuis {args.model_dir}")
        classifier = PlantDiseaseClassifier.load_model(Path(args.model_dir) / 'config.json')
    else:
        print("\nInitialisation d'un nouveau modèle...")
        classifier = PlantDiseaseClassifier(config)
    
    # Entraînement
    if config.mode == 'train':
        print("\nDébut de l'entraînement...")
        history = classifier.train(train_generator, val_generator)
        
        # Visualisation des courbes d'apprentissage
        plot_training_history(history, config.plots_dir)
    
    # Évaluation
    if config.mode in ['evaluate', 'train']:
        print("\nÉvaluation du modèle...")
        metrics = classifier.evaluate(test_generator)
        
        # Enregistrement des métriques
        metrics_file = config.output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Affichage des métriques
        print("\nMétriques d'évaluation:")
        print(f"- Précision: {metrics['accuracy']:.4f}")
        if 'auc' in metrics:
            print(f"- AUC: {metrics['auc']:.4f}")
        print(f"- F1-score (macro): {metrics['f1_macro']:.4f}")
        print(f"- F1-score (weighted): {metrics['f1_weighted']:.4f}")
        print(f"- Temps d'inférence: {metrics['inference_time']:.2f}s")
    
    # Explication des prédictions
    if config.mode == 'explain':
        print("\nGénération des explications...")
        # Sélection de quelques échantillons
        num_samples = min(5, test_generator.samples)
        test_generator.reset()
        X_sample, _ = next(test_generator)
        X_sample = X_sample[:num_samples]
        
        # Génération des explications
        classifier.explain_predictions(X_sample, class_names)
        
        print("\nExplications générées avec succès!")

if __name__ == "__main__":
    main()
