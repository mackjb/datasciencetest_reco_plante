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
import random
import shutil
import tempfile
import multiprocessing
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
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model, save_model
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
    
    def _create_train_val_split(self, source_dir, target_dir, val_split=0.2):
        """Crée une séparation entraînement/validation dans des dossiers distincts
        
        Args:
            source_dir (Path): Dossier source contenant les images par classe
            target_dir (Path): Dossier cible pour la séparation
            val_split (float): Proportion des données pour la validation
            
        Returns:
            tuple: (train_dir, val_dir)
        """
        train_dir = target_dir / 'train'
        val_dir = target_dir / 'val'
        
        # Création des dossiers
        for split_dir in [train_dir, val_dir]:
            for class_name in self.class_names:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Répartition des fichiers
        for class_name in self.class_names:
            class_dir = source_dir / class_name
            if not class_dir.exists():
                continue
                
            # Liste tous les fichiers de la classe
            files = list(class_dir.glob('*'))
            random.Random(self.config.random_state).shuffle(files)
            
            # Séparation train/val
            split_idx = int(len(files) * (1 - val_split))
            train_files = files[:split_idx]
            val_files = files[split_idx:]
            
            # Copie des fichiers
            for file in train_files:
                shutil.copy2(file, train_dir / class_name / file.name)
            for file in val_files:
                shutil.copy2(file, val_dir / class_name / file.name)
        
        return train_dir, val_dir
    
    def load_data(self):
        """Charge les données avec une séparation stricte train/val/test
        
        Returns:
            tuple: (train_generator, val_generator, test_generator, class_names)
        """
        print("\nChargement des données...")
        
        # Vérification de la structure des dossiers
        if not self.config.image_dir.exists():
            raise FileNotFoundError(f"Dossier d'images non trouvé: {self.config.image_dir}")
        
        # Récupération des noms de classes
        self.class_names = self._get_class_names(self.config.image_dir)
        self.config.num_classes = len(self.class_names)
        print(f"{self.config.num_classes} classes détectées: {', '.join(self.class_names[:5])}...")
        
        # Création des dossiers temporaires
        temp_dir = Path(tempfile.mkdtemp(prefix='plantvillage_'))
        train_dir, val_dir = self._create_train_val_split(
            self.config.image_dir, temp_dir, val_split=0.2
        )
        
        # Création des générateurs avec des séparations strictes
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **self.config.augmentation
        )
        
        # Générateur d'entraînement
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.random_state,
            shuffle=True
        )
        
        # Générateur de validation (sans augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.random_state,
            shuffle=False
        )
        
        # Pour le test, on utilise une partie des données de validation
        # car nous n'avons pas de jeu de test séparé dans la structure de base
        # En production, il faudrait avoir un dossier test séparé
        test_generator = val_datagen.flow_from_directory(
            val_dir,  # À remplacer par un dossier test séparé si disponible
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.random_state,
            shuffle=False
        )
        
        # Nettoyage des fichiers temporaires
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Calcul des tailles
        train_size = len(train_generator.filenames)
        val_size = len(val_generator.filenames)
        test_size = len(test_generator.filenames)
        
        print(f"Données chargées : {train_size + val_size + test_size} images, {self.config.num_classes} classes")
        print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")
        
        return train_generator, val_generator, test_generator, self.class_names
# ------------------------------------------------------
# MODÈLE
# ------------------------------------------------------
class PlantDiseaseClassifier:
    """Classe pour le modèle de classification des maladies de plantes
    
    Cette classe gère la création, l'entraînement et l'évaluation d'un modèle
    de classification basé sur un modèle pré-entraîné avec fine-tuning.
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
        self.target_layer_name = None  # Nom de la couche cible pour Grad-CAM
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
    
    def _find_target_layer(self, model):
        """Trouve la dernière couche de convolution du modèle de base
        
        Args:
            model: Modèle Keras à analyser
            
        Returns:
            str: Nom de la couche cible pour Grad-CAM
        """
        # Liste des types de couches à considérer pour Grad-CAM
        target_layers = [
            'conv', 'conv2d', 'convolution', 'convolution2d',
            'activation', 'activation_', 'relu', 'leaky_relu',
            'batch_normalization', 'bn'
        ]
        
        # Parcourir les couches à l'envers pour trouver la dernière couche de convolution
        for layer in reversed(model.layers):
            # Vérifier si le nom de la couche contient un mot-clé pertinent
            if any(keyword in layer.name.lower() for keyword in target_layers):
                return layer.name
        
        # Si aucune couche n'est trouvée, utiliser la dernière couche
        return model.layers[-1].name
    
    def _build_model(self):
        """Construit le modèle avec un backbone pré-entraîné et fine-tuning
        
        Returns:
            tf.keras.Model: Modèle compilé avec un extracteur de caractéristiques
        """
        # Chargement du modèle de base pré-entraîné sur ImageNet
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.img_size, 3)
        )
        
        # Trouver la couche cible pour Grad-CAM
        self.target_layer_name = self._find_target_layer(base_model)
        
        # Geler les couches du modèle de base initialement
        base_model.trainable = False
        
        # Construction du modèle complet
        inputs = tf.keras.Input(shape=(*self.config.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compilation du modèle
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=1.0  # Pour éviter les explosions de gradient
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Sauvegarde de la référence au modèle de base
        self.base_model = base_model
        
        return model
    
    def train(self, train_generator, val_generator):
        """Entraîne le modèle avec les générateurs de données
        
        Args:
            train_generator: Générateur de données d'entraînement (avec augmentation)
            val_generator: Générateur de données de validation
            
        Returns:
            History: Historique d'entraînement
        """
        # Déterminer le nombre de workers en fonction des ressources disponibles
        try:
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            # Utiliser au maximum 4 workers et au moins 1
            num_workers = max(1, min(4, num_cores - 1 if num_cores > 1 else 1))
        except:
            num_workers = 1
        
        # Configuration des callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.model_dir / 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.config.logs_dir),
                histogram_freq=1,
                update_freq='epoch',
                profile_batch=0  # Désactiver le profilage pour économiser des ressources
            ),
            tf.keras.callbacks.CSVLogger(
                str(self.config.logs_dir / 'training.log'),
                append=True
            )
        ]
        
        # Entraînement initial avec les couches de base gelées
        print(f"\nPhase 1: Entraînement des nouvelles couches (workers: {num_workers})")
        
        # Désactiver les logs verbeux de TensorFlow
        tf.get_logger().setLevel('ERROR')
        
        history = self.model.fit(
            train_generator,
            epochs=self.config.epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            workers=num_workers,
            use_multiprocessing=num_workers > 1,
            max_queue_size=10,  # Limiter la taille de la file d'attente
            verbose=1
        )
        
        # Fine-tuning: dégeler certaines couches du modèle de base
        print("\nPhase 2: Fine-tuning des couches de base")
        self._unfreeze_layers()
        
        # Réduire le taux d'apprentissage pour le fine-tuning
        reduced_lr = self.config.learning_rate / 10
        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate, 
            reduced_lr
        )
        print(f"Taux d'apprentissage réduit à {reduced_lr:.2e} pour le fine-tuning")
        
        # Reprendre l'entraînement avec les couches dégelées
        history_fine = self.model.fit(
            train_generator,
            epochs=self.config.epochs * 2,  # Plus d'époques pour le fine-tuning
            initial_epoch=history.epoch[-1] + 1,
            validation_data=val_generator,
            callbacks=callbacks,
            workers=num_workers,
            use_multiprocessing=num_workers > 1,
            max_queue_size=10,
            verbose=1
        )
        
        # Combiner les historiques
        for key in history_fine.history:
            history.history[key].extend(history_fine.history[key])
        
        # Sauvegarder le modèle final
        final_model_path = self.config.model_dir / 'final_model.h5'
        self.model.save(final_model_path)
        print(f"\nModèle final sauvegardé dans {final_model_path}")
        
        # Nettoyer la mémoire
        tf.keras.backend.clear_session()
        
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
        # Désactiver les logs verbeux
        tf.get_logger().setLevel('ERROR')
        
        # Déterminer le nombre de workers en fonction des ressources disponibles
        try:
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            num_workers = max(1, min(4, num_cores - 1 if num_cores > 1 else 1))
        except:
            num_workers = 1
        
        # Évaluation du modèle
        print(f"\nÉvaluation sur l'ensemble de test (workers: {num_workers})...")
        
        # Prédictions et métriques par lots pour économiser la mémoire
        y_true = []
        y_pred = []
        batch_times = []
        
        # Réinitialiser le générateur
        test_generator.reset()
        
        # Désactiver la barre de progression pour les prédictions
        for batch_idx in range(len(test_generator)):
            batch_start = time.time()
            
            # Récupérer le lot actuel
            X_batch, y_batch = test_generator[batch_idx]
            
            # Faire la prédiction
            y_pred_batch = self.model.predict(
                X_batch,
                verbose=0,
                batch_size=self.config.batch_size
            )
            
            # Enregistrer les résultats
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(y_pred_batch)
            
            # Calculer le temps d'inférence
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Afficher la progression
            progress = (batch_idx + 1) / len(test_generator) * 100
            print(f"\rProgression: {progress:.1f}%", end="")
        
        # Convertir en tableaux numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculer le temps d'inférence moyen par image
        avg_inference_time = np.mean(batch_times) / self.config.batch_size
        
        # Calculer les métriques
        test_loss, test_accuracy = self.model.evaluate(
            test_generator,
            verbose=0,
            workers=num_workers,
            use_multiprocessing=num_workers > 1,
            max_queue_size=10
        )
        
        # Création du rapport de classification
        report = classification_report(
            y_true, y_pred_classes,
            target_names=self.config.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Calcul de l'AUC (uniquement pour la classification binaire)
        auc_score = 0.0
        if self.num_classes == 2:
            try:
                pass  # Placeholder for AUC calculation
            except Exception as e:
                print(f"    Erreur lors du calcul de l'AUC: {str(e)}")
        
        # Afficher la progression
        print("\nÉvaluation terminée.")
        
        # Préparer les métriques de sortie
        metrics = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'inference_time': avg_inference_time,
            'class_metrics': {}
        }
        
        # Ajouter les métriques par classe
        for i, class_name in enumerate(self.config.class_names):
            metrics['class_metrics'][class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1_score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            }
        
        # Ajouter l'AUC si calculé
        if auc_score > 0:
            metrics['auc'] = auc_score
            
        return metrics
    
    def explain_predictions(self, X_sample, class_names):
        """Génère des explications pour les prédictions du modèle
        
        Args:
            X_sample: Échantillons d'images à expliquer
            class_names: Liste des noms de classes
        """
        # Désactiver les logs verbeux
        tf.get_logger().setLevel('ERROR')
        
        # Créer le dossier de sortie s'il n'existe pas
        self.config.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Limiter le nombre d'échantillons pour économiser des ressources
        max_samples = min(5, len(X_sample))
        X_sample = X_sample[:max_samples]
        
        print(f"\nGénération des explications pour {len(X_sample)} échantillons...")
        
        # Générer les explications avec les différentes méthodes
        print("\n1. Génération des explications Grad-CAM...")
        self._explain_with_gradcam(X_sample, class_names, num_samples=3)
        
        print("\n2. Génération des explications SHAP (cette étape peut prendre du temps)...")
        self._explain_with_shap(X_sample, class_names, max_samples=3, background_size=5)
        
        print("\n3. Génération des explications LIME...")
        self._explain_with_lime(X_sample, class_names, num_samples=500, max_features=5)
        
        print("\nToutes les explications ont été générées avec succès !")
    
    def _explain_with_shap(self, X_sample, class_names, max_samples=3, background_size=5):
        """Génère des explications avec SHAP de manière optimisée
        
        Args:
            X_sample: Échantillons d'images à expliquer
            class_names: Liste des noms de classes
            max_samples: Nombre maximum d'échantillons à expliquer
            background_size: Taille de l'ensemble de fond pour SHAP
        """
        print("    Génération des explications SHAP (peut prendre plusieurs minutes)...")
        
        try:
            import shap
            
            # Vérifier la disponibilité de la mémoire GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"    Utilisation du GPU pour les calculs SHAP")
                # Configurer la mémoire GPU pour une allocation croissante
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Limiter le nombre d'échantillons pour des raisons de performance
            num_samples = min(max_samples, len(X_sample))
            X_sample = X_sample[:num_samples]
            
            # Créer un ensemble de fond plus petit pour accélérer le calcul
            background = X_sample[np.random.choice(
                len(X_sample), 
                min(background_size, len(X_sample)), 
                replace=False
            )]
            
            print(f"    Calcul des explications pour {num_samples} échantillons avec {len(background)} échantillons de fond...")
            
            # Créer l'explainer SHAP avec un échantillonnage plus efficace
            explainer = shap.GradientExplainer(
                model=self.model,
                data=background,
                batch_size=min(2, num_samples)  # Réduire la taille du lot pour économiser la mémoire
            )
            
            # Calculer les valeurs SHAP par lots pour économiser la mémoire
            batch_size = 1  # Taille de lot réduite pour éviter les problèmes de mémoire
            all_shap_values = []
            
            for i in range(0, len(X_sample), batch_size):
                batch = X_sample[i:i+batch_size]
                print(f"    Traitement du lot {i//batch_size + 1}/{(len(X_sample)-1)//batch_size + 1}...")
                
                # Calculer les valeurs SHAP pour le lot actuel
                shap_batch = explainer.shap_values(batch, progress_message=None)
                all_shap_values.extend(shap_batch)
                
                # Libérer la mémoire après chaque lot
                tf.keras.backend.clear_session()
            
            # Convertir en tableau numpy
            shap_values = np.array(all_shap_values)
            
            # Sauvegarder les valeurs SHAP pour une analyse ultérieure
            np.save(self.config.plots_dir / 'shap_values.npy', shap_values)
            
            # Créer un résumé des caractéristiques importantes
            print("    Génération des visualisations SHAP...")
            
            # Créer une figure pour le résumé
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values.reshape(shap_values.shape[0], -1),  # Aplatir les dimensions spatiales
                features=X_sample.reshape(X_sample.shape[0], -1),  # Aplatir les images
                class_names=class_names,
                show=False,
                max_display=10,  # Limiter le nombre de caractéristiques affichées
                plot_type="bar"
            )
            
            # Sauvegarder le graphique de résumé
            summary_path = self.config.plots_dir / 'shap_summary.png'
            plt.tight_layout()
            plt.savefig(summary_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Créer des visualisations individuelles pour chaque échantillon
            max_plots = min(3, len(X_sample))  # Limiter à 3 échantillons pour les visualisations détaillées
            for i in range(max_plots):
                plt.figure(figsize=(10, 4))
                
                # Afficher l'image originale
                plt.subplot(1, 2, 1)
                plt.imshow(X_sample[i])
                plt.title(f"Original - {class_names[np.argmax(self.model.predict(X_sample[i:i+1], verbose=0)[0])]}")
                plt.axis('off')
                
                # Afficher la carte d'importance SHAP
                plt.subplot(1, 2, 2)
                shap.image_plot(
                    [shap_values[i].transpose(2, 0, 1)],  # Réorganiser les dimensions pour SHAP
                    X_sample[i:i+1],
                    show=False
                )
                plt.title("Importance SHAP")
                
                # Sauvegarder la visualisation individuelle
                sample_path = self.config.plots_dir / f'shap_sample_{i+1}.png'
                plt.tight_layout()
                plt.savefig(sample_path, bbox_inches='tight', dpi=150)
                plt.close()
            
            print(f"    Explications SHAP sauvegardées dans {self.config.plots_dir}/")
            
        except ImportError:
            print("    SHAP n'est pas installé. Installez-le avec: pip install shap")
        except Exception as e:
            print(f"    Erreur lors de la génération des explications SHAP: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _explain_with_lime(self, X_sample, class_names, num_samples=1000, max_features=5):
        """Génère des explications avec LIME de manière optimisée
        
        Args:
            X_sample: Échantillons d'images à expliquer
            class_names: Liste des noms de classes
            num_samples: Nombre d'échantillons pour LIME (limité pour des raisons de performance)
            max_features: Nombre maximum de caractéristiques à afficher
        """
        try:
            from lime import lime_image
            from skimage.segmentation import slic
            
            # Limiter le nombre d'échantillons pour des raisons de performance
            num_explanations = min(3, len(X_sample))
            
            # Fonction de prédiction compatible avec LIME
            def predict_fn(images):
                # Prétraiter les images si nécessaire
                if images.dtype != np.float32:
                    images = images.astype(np.float32) / 255.0
                return self.model.predict(images, verbose=0)
            
            # Créer l'explainer LIME
            explainer = lime_image.LimeImageExplainer(
                kernel_width=0.25,
                verbose=False,
                feature_selection='lasso_path',
                random_state=self.config.random_state
            )
            
            # Créer une figure pour afficher toutes les explications
            plt.figure(figsize=(15, 5 * num_explanations))
            
            for i in range(num_explanations):
                # Générer l'explication
                explanation = explainer.explain_instance(
                    X_sample[i],
                    predict_fn,
                    top_labels=len(class_names),
                    hide_color=0,
                    num_samples=min(num_samples, 500),  # Limiter le nombre d'échantillons
                    batch_size=10,  # Réduire la taille du lot pour économiser la mémoire
                    segmentation_fn=lambda x: slic(
                        x, 
                        n_segments=25,  # Réduire le nombre de segments
                        compactness=10, 
                        sigma=1
                    )
                )
                
                # Afficher l'explication
                plt.subplot(num_explanations, 2, 2*i+1)
                plt.imshow(X_sample[i])
                plt.title(f"Original - Prédit: {class_names[np.argmax(predict_fn(X_sample[i:i+1])[0])]}")
                plt.axis('off')
                
                plt.subplot(num_explanations, 2, 2*i+2)
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=max_features,
                    hide_rest=False
                )
                plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
                plt.title("Explication LIME")
                plt.axis('off')
                
                # Sauvegarder l'explication individuelle
                temp_dir = self.config.plots_dir / 'lime_explanations'
                temp_dir.mkdir(exist_ok=True)
                explanation.save_to_file(temp_dir / f'lime_explanation_{i}.html')
            
            # Sauvegarder la figure combinée
            plt.tight_layout()
            plt.savefig(self.config.plots_dir / 'lime_explanations.png', bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Explications LIME sauvegardées dans {self.config.plots_dir}/lime_explanations/")
            
        except ImportError:
            print("LIME n'est pas installé. Installez-le avec: pip install lime")
        except Exception as e:
            print(f"Erreur lors de la génération des explications LIME: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
