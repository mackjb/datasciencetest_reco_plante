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
    classification_report,
    roc_auc_score
)

# Utilitaires
from .utils import Logger, ensure_dir, clear_memory, copy_files

# Désactiver les avertissements inutiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class Config:
    """Configuration du modèle et des chemins"""
    def __init__(self):
        # Paramètres du modèle
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 20
        self.initial_learning_rate = 1e-4
        self.fine_tune_learning_rate = 1e-5
        self.dropout_rate = 0.3
        self.fine_tune_layers = 50
        self.num_classes = None
        self.auc_multi_class = 'ovr'
        self.max_explanation_samples = 3
        
        # Chemins des données
        self.data_path = Path("dataset/plantvillage/")
        self.image_dir = self.data_path / "images"
        self.label_csv = self.data_path / "labels.csv"
        
        # Dossiers de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = ensure_dir(Path(f"results/plantvillage_{timestamp}"))
        self.model_dir = ensure_dir(self.output_dir / "models")
        self.logs_dir = ensure_dir(self.output_dir / "logs")
        self.plots_dir = ensure_dir(self.output_dir / "plots")
        
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
        
        # Paramètres de reproductibilité
        self.random_state = 42
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def save(self, filepath=None):
        """Sauvegarde la configuration dans un fichier JSON"""
        if filepath is None:
            filepath = self.output_dir / "config.json"
        
        config_dict = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'initial_learning_rate': self.initial_learning_rate,
            'fine_tune_learning_rate': self.fine_tune_learning_rate,
            'dropout_rate': self.dropout_rate,
            'fine_tune_layers': self.fine_tune_layers,
            'num_classes': self.num_classes,
            'auc_multi_class': self.auc_multi_class,
            'max_explanation_samples': self.max_explanation_samples,
            'random_state': self.random_state
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Charge une configuration depuis un fichier JSON"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

class PlantVillageDataLoader:
    """Classe pour charger et prétraiter les données PlantVillage"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger()
        self.class_names = []
    
    def _get_class_names(self, directory):
        """Récupère la liste des noms de classes depuis la structure des dossiers"""
        return sorted([d for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d))])
    
    def create_data_splits(self, source_dir, val_split=0.15, test_split=0.15):
        """Crée les splits train/val/test dans des dossiers temporaires"""
        splits = ['train', 'val', 'test']
        split_dirs = {split: str(ensure_dir(Path(tempfile.mkdtemp()) / split)) 
                     for split in splits}
        
        try:
            for class_name in self._get_class_names(source_dir):
                class_dir = os.path.join(source_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                random.shuffle(files)
                
                # Calculer les indices de séparation
                val_idx = int(len(files) * (1 - val_split - test_split))
                test_idx = int(len(files) * (1 - test_split))
                
                # Copier les fichiers
                copy_files(files[:val_idx], class_dir, split_dirs['train'], class_name)
                copy_files(files[val_idx:test_idx], class_dir, split_dirs['val'], class_name)
                copy_files(files[test_idx:], class_dir, split_dirs['test'], class_name)
                
            return split_dirs
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des splits: {str(e)}")
            raise
    
    def load_data(self):
        """Charge les données avec une séparation stricte train/val/test"""
        try:
            # Créer les dossiers temporaires pour les splits
            split_dirs = self.create_data_splits(
                str(self.config.image_dir),
                val_split=0.15,
                test_split=0.15
            )
            
            # Récupérer les noms des classes
            self.class_names = self._get_class_names(split_dirs['train'])
            self.config.num_classes = len(self.class_names)
            
            # Créer les générateurs de données
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                **self.config.augmentation
            )
            
            val_test_datagen = ImageDataGenerator(rescale=1./255)
            
            # Charger les données
            train_generator = train_datagen.flow_from_directory(
                split_dirs['train'],
                target_size=self.config.img_size,
                batch_size=self.config.batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=self.config.random_state
            )
            
            val_generator = val_test_datagen.flow_from_directory(
                split_dirs['val'],
                target_size=self.config.img_size,
                batch_size=self.config.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            test_generator = val_test_datagen.flow_from_directory(
                split_dirs['test'],
                target_size=self.config.img_size,
                batch_size=self.config.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            return train_generator, val_generator, test_generator, self.class_names
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise

class PlantDiseaseClassifier:
    """Classe pour le modèle de classification des maladies de plantes"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger()
        self.model = None
        self.base_model = None
        self.target_layer_name = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle en créant un nouveau ou en chargeant un existant"""
        if self.config.model_path.exists():
            self._load_existing_model()
        else:
            self._build_new_model()
    
    def _build_new_model(self):
        """Construit un nouveau modèle à partir de zéro"""
        self.logger.info("Construction d'un nouveau modèle...")
        self.model = self._build_model()
        self._compile_model()
    
    def _load_existing_model(self):
        """Charge un modèle existant depuis le disque"""
        try:
            self.logger.info(f"Chargement du modèle depuis {self.config.model_path}...")
            self.model = load_model(self.config.model_path)
            self.base_model = self.model.layers[0]
            self.target_layer_name = self._find_target_layer(self.base_model)
            self.logger.info("Modèle chargé avec succès")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    
    def _find_target_layer(self, model):
        """Trouve la dernière couche de convolution pour Grad-CAM"""
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Aucune couche de convolution trouvée dans le modèle")
    
    def _build_model(self):
        """Construit le modèle avec un backbone pré-entraîné"""
        # Charger le modèle de base pré-entraîné
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.img_size, 3)
        )
        
        # Geler les couches du modèle de base
        for layer in base_model.layers:
            layer.trainable = False
        
        # Ajouter des couches personnalisées
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(self.config.dropout_rate)(x)
        predictions = Dense(self.config.num_classes, activation='softmax')(x)
        
        # Créer le modèle final
        model = Model(inputs=base_model.input, outputs=predictions)
        self.base_model = base_model
        self.target_layer_name = self._find_target_layer(base_model)
        
        return model
    
    def _compile_model(self):
        """Compile le modèle avec les paramètres d'optimisation"""
        optimizer = Adam(learning_rate=self.config.initial_learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _unfreeze_layers(self):
        """Dégèle progressivement les couches pour le fine-tuning"""
        if not self.base_model:
            raise ValueError("Le modèle de base n'est pas initialisé")
            
        # Ne dégeler que les N dernières couches
        num_layers = len(self.base_model.layers)
        trainable_start = num_layers - self.config.fine_tune_layers
        
        for i, layer in enumerate(self.base_model.layers):
            layer.trainable = (i >= trainable_start)
            if layer.trainable and hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)
        
        # Recompiler avec un taux d'apprentissage plus faible
        optimizer = Adam(learning_rate=self.config.fine_tune_learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info(f"{self.config.fine_tune_layers} couches dégelées pour le fine-tuning")
    
    def train(self, train_generator, val_generator):
        """Entraîne le modèle avec les données fournies"""
        try:
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                ),
                ModelCheckpoint(
                    filepath=str(self.config.model_path),
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                ),
                TensorBoard(
                    log_dir=str(self.config.logs_dir),
                    histogram_freq=1
                )
            ]
            
            # Entraînement initial
            self.logger.info("Début de l'entraînement initial...")
            history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.epochs,
                callbacks=callbacks,
                workers=4,
                use_multiprocessing=True
            )
            
            # Fine-tuning
            self.logger.info("Début du fine-tuning...")
            self._unfreeze_layers()
            
            history_fine = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.epochs,
                initial_epoch=history.epoch[-1],
                callbacks=callbacks,
                workers=4,
                use_multiprocessing=True
            )
            
            # Fusionner les historiques
            for key in history.history.keys():
                history.history[key].extend(history_fine.history[key])
            
            return history
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            raise
    
    def evaluate(self, test_generator):
        """Évalue le modèle sur l'ensemble de test"""
        try:
            self.logger.info("Évaluation du modèle...")
            
            # Prédictions
            y_pred = self.model.predict(
                test_generator,
                workers=4,
                use_multiprocessing=True
            )
            y_true = test_generator.classes
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Métriques
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred_classes),
                'f1_macro': f1_score(y_true, y_pred_classes, average='macro'),
                'f1_weighted': f1_score(y_true, y_pred_classes, average='weighted'),
                'auc': self.calculate_auc(y_true, y_pred)
            }
            
            # Matrice de confusion
            cm = confusion_matrix(y_true, y_pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.savefig(str(self.config.plots_dir / 'confusion_matrix.png'))
            plt.close()
            
            # Rapport de classification
            report = classification_report(
                y_true,
                y_pred_classes,
                target_names=test_generator.class_indices.keys(),
                output_dict=True
            )
            
            # Sauvegarder les métriques
            metrics.update({
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            })
            
            with open(str(self.config.metrics_file), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation: {str(e)}")
            raise
    
    def calculate_auc(self, y_true, y_pred) -> Optional[float]:
        """Calcule l'AUC pour la classification multi-classes"""
        from sklearn.metrics import roc_auc_score
        from tensorflow.keras.utils import to_categorical
        
        try:
            if self.config.num_classes == 2:
                return roc_auc_score(y_true, y_pred[:, 1])
            else:
                return roc_auc_score(
                    to_categorical(y_true, num_classes=self.config.num_classes),
                    y_pred,
                    multi_class=self.config.auc_multi_class
                )
        except Exception as e:
            self.logger.warning(f"Impossible de calculer l'AUC: {str(e)}")
            return None
    
    def explain_predictions(self, X_sample, class_names):
        """Génère des explications pour les prédictions"""
        try:
            clear_memory()
            
            # Limiter le nombre d'échantillons
            max_samples = min(self.config.max_explanation_samples, len(X_sample))
            X_sample = X_sample[:max_samples]
            
            # Générer les explications
            self._explain_with_gradcam(X_sample, class_names)
            self._explain_with_shap(X_sample, class_names)
            self._explain_with_lime(X_sample, class_names)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des explications: {str(e)}")
            raise
    
    def _explain_with_gradcam(self, X_sample, class_names, num_samples=3):
        """Génère des explications avec Grad-CAM"""
        try:
            if self.target_layer_name is None:
                self.logger.warning("Impossible de trouver une couche cible pour Grad-CAM")
                return
                
            # Limiter le nombre d'échantillons
            num_samples = min(num_samples, len(X_sample))
            
            # Créer un modèle pour les activations
            grad_model = Model(
                [self.model.inputs],
                [self.model.get_layer(self.target_layer_name).output, self.model.output]
            )
            
            # Préparer la figure
            plt.figure(figsize=(15, 5 * num_samples))
            
            for i in range(num_samples):
                # Prédiction
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(np.array([X_sample[i]]))
                    class_idx = tf.argmax(predictions[0])
                    loss = predictions[:, class_idx]
                
                # Calcul des gradients
                grads = tape.gradient(loss, conv_outputs)[0]
                weights = tf.reduce_mean(grads, axis=(0, 1))
                
                # Construction de la heatmap
                cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
                for j, w in enumerate(weights):
                    cam += w * conv_outputs[0, :, :, j]
                
                cam = np.maximum(cam, 0)
                cam = cv2.resize(cam, self.config.img_size)
                cam = cam / np.max(cam)
                
                # Visualisation
                plt.subplot(num_samples, 2, 2*i+1)
                plt.imshow(X_sample[i])
                plt.title(f"Original - Prédit: {class_names[np.argmax(predictions[0])]}")
                plt.axis('off')
                
                plt.subplot(num_samples, 2, 2*i+2)
                plt.imshow(X_sample[i])
                plt.imshow(cam, cmap='jet', alpha=0.5)
                plt.title("Grad-CAM")
                plt.axis('off')
            
            # Sauvegarder la figure
            plt.tight_layout()
            plt.savefig(str(self.config.plots_dir / 'gradcam_explanations.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            self.logger.info("Explications Grad-CAM générées avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des explications Grad-CAM: {str(e)}")
            raise
    
    def _explain_with_shap(self, X_sample, class_names, max_samples=3, background_size=5):
        """Génère des explications avec SHAP"""
        try:
            import shap
            clear_memory()
            
            max_samples = min(max_samples, self.config.max_explanation_samples, len(X_sample))
            X_sample = X_sample[:max_samples]
            
            self.logger.info(f"Génération des explications SHAP pour {max_samples} échantillons...")
            
            # Créer un ensemble de fond plus petit
            background = X_sample[np.random.choice(
                len(X_sample), 
                min(background_size, len(X_sample)), 
                replace=False
            )]
            
            # Créer l'explainer SHAP
            explainer = shap.GradientExplainer(
                model=self.model,
                data=background,
                batch_size=min(2, len(X_sample))
            )
            
            # Calculer les valeurs SHAP par lots
            batch_size = 1
            all_shap_values = []
            
            for i in range(0, len(X_sample), batch_size):
                batch = X_sample[i:i+batch_size]
                shap_batch = explainer.shap_values(batch, progress_message=None)
                all_shap_values.extend(shap_batch)
                clear_memory()
            
            # Convertir en tableau numpy
            shap_values = np.array(all_shap_values)
            
            # Sauvegarder les valeurs SHAP
            np.save(str(self.config.plots_dir / 'shap_values.npy'), shap_values)
            
            # Visualisation du résumé
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values.reshape(shap_values.shape[0], -1),
                features=X_sample.reshape(X_sample.shape[0], -1),
                class_names=class_names,
                show=False,
                max_display=10,
                plot_type="bar"
            )
            plt.tight_layout()
            plt.savefig(str(self.config.plots_dir / 'shap_summary.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            # Visualisations individuelles
            for i in range(min(3, len(X_sample))):
                plt.figure(figsize=(10, 4))
                
                # Image originale
                plt.subplot(1, 2, 1)
                plt.imshow(X_sample[i])
                plt.title(f"Original - {class_names[np.argmax(self.model.predict(X_sample[i:i+1], verbose=0)[0])]}")
                plt.axis('off')
                
                # Carte d'importance SHAP
                plt.subplot(1, 2, 2)
                shap.image_plot(
                    [shap_values[i].transpose(2, 0, 1)],
                    X_sample[i:i+1],
                    show=False
                )
                plt.title("Importance SHAP")
                
                # Sauvegarder
                plt.tight_layout()
                plt.savefig(str(self.config.plots_dir / f'shap_sample_{i+1}.png'), 
                           bbox_inches='tight', dpi=150)
                plt.close()
            
            self.logger.info("Explications SHAP générées avec succès")
            
        except ImportError:
            self.logger.warning("SHAP n'est pas installé. Installez-le avec: pip install shap")
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des explications SHAP: {str(e)}")
            raise
    
    def _explain_with_lime(self, X_sample, class_names, num_samples=1000, max_features=5):
        """Génère des explications avec LIME"""
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries
            clear_memory()
            
            num_explanations = min(3, self.config.max_explanation_samples, len(X_sample))
            
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
                    num_samples=min(num_samples, 500),
                    batch_size=10,
                    segmentation_fn=lambda x: slic(
                        x, 
                        n_segments=25,
                        compactness=10, 
                        sigma=1
                    )
                )
                
                # Afficher l'image originale
                plt.subplot(num_explanations, 2, 2*i+1)
                plt.imshow(X_sample[i])
                plt.title(f"Original - Prédit: {class_names[np.argmax(predict_fn(X_sample[i:i+1])[0])]}")
                plt.axis('off')
                
                # Afficher l'explication LIME
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
                lime_dir = ensure_dir(self.config.plots_dir / 'lime_explanations')
                explanation.save_to_file(str(lime_dir / f'lime_explanation_{i}.html'))
            
            # Sauvegarder la figure combinée
            plt.tight_layout()
            plt.savefig(str(self.config.plots_dir / 'lime_explanations.png'), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            self.logger.info("Explications LIME générées avec succès")
            
        except ImportError:
            self.logger.warning("LIME n'est pas installé. Installez-le avec: pip install lime")
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des explications LIME: {str(e)}")
            raise
    
    def save_model(self, model_name='best_model'):
        """Sauvegarde le modèle et la configuration"""
        try:
            # Sauvegarder le modèle
            model_path = self.config.model_dir / f"{model_name}.h5"
            self.model.save(str(model_path))
            
            # Sauvegarder la configuration
            config_path = self.config.model_dir / f"{model_name}_config.json"
            self.config.save(config_path)
            
            self.logger.info(f"Modèle sauvegardé dans {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, config_path):
        """Charge un modèle existant à partir d'un fichier de configuration"""
        try:
            config = Config.load(config_path)
            classifier = cls(config)
            return classifier
        except Exception as e:
            Logger().error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

def main():
    """Fonction principale pour exécuter le pipeline complet"""
    try:
        # Configuration
        config = Config()
        logger = Logger(str(config.output_dir / 'pipeline.log'))
        
        # 1. Chargement des données
        logger.info("Chargement des données...")
        data_loader = PlantVillageDataLoader(config)
        train_generator, val_generator, test_generator, class_names = data_loader.load_data()
        
        # 2. Initialisation du modèle
        logger.info("Initialisation du modèle...")
        model = PlantDiseaseClassifier(config)
        
        # 3. Entraînement
        logger.info("Début de l'entraînement...")
        history = model.train(train_generator, val_generator)
        
        # 4. Évaluation
        logger.info("Évaluation du modèle...")
        metrics = model.evaluate(test_generator)
        logger.info(f"Résultats de l'évaluation: {metrics}")
        
        # 5. Explications (sur un sous-ensemble des données de test)
        logger.info("Génération des explications...")
        num_samples = min(5, len(test_generator.filenames))
        sample_indices = np.random.choice(len(test_generator.filenames), num_samples, replace=False)
        X_sample = np.array([test_generator[i][0][0] for i in sample_indices])
        model.explain_predictions(X_sample, class_names)
        
        logger.info("Pipeline terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur dans le pipeline principal: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
