#!/usr/bin/env python3
# Script de nettoyage pour supprimer les fichiers temporaires et inutiles

import os
import glob
import shutil
from datetime import datetime, timedelta

def clean_old_files(directory, days=1, extensions=None):
    """
    Supprime les fichiers plus vieux que 'days' jours avec les extensions spécifiées
    """
    if extensions is None:
        extensions = ['.pkl', '.pyc', '.log', '.tmp']
    
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    
    for ext in extensions:
        pattern = os.path.join(directory, f'**/*{ext}')
        for filepath in glob.glob(pattern, recursive=True):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff:
                try:
                    os.remove(filepath)
                    print(f"Supprimé: {filepath}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {filepath}: {e}")

def organize_directory(directory):
    """Organise les fichiers dans des dossiers par type"""
    file_types = {
        'images': ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
        'models': ['.joblib', '.h5', '.hdf5', '.onnx'],
        'data': ['.csv', '.json', '.parquet', '.feather'],
        'logs': ['.log'],
        'docs': ['.txt', '.md', '.pdf']
    }
    
    # Créer les dossiers s'ils n'existent pas
    for folder in file_types.keys():
        os.makedirs(os.path.join(directory, folder), exist_ok=True)
    
    # Déplacer les fichiers dans les dossiers appropriés
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Ignorer les dossiers et les fichiers cachés
        if os.path.isdir(item_path) or item.startswith('.'):
            continue
            
        # Obtenir l'extension du fichier
        _, ext = os.path.splitext(item)
        ext = ext.lower()
        
        # Trouver le dossier de destination
        dest_folder = None
        for folder, extensions in file_types.items():
            if ext in extensions:
                dest_folder = folder
                break
        
        # Déplacer le fichier
        if dest_folder:
            dest_path = os.path.join(directory, dest_folder, item)
            try:
                shutil.move(item_path, dest_path)
                print(f"Déplacé: {item} -> {dest_folder}/")
            except Exception as e:
                print(f"Erreur lors du déplacement de {item}: {e}")

def main():
    # Nettoyer les anciens fichiers
    print("Nettoyage des anciens fichiers...")
    clean_old_files('.', days=1)
    
    # Organiser le répertoire
    print("\nOrganisation du répertoire...")
    organize_directory('.')
    
    print("\nNettoyage terminé avec succès!")

if __name__ == "__main__":
    main()
