import os
import logging
import tempfile
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Logger:
    """Classe de gestion des logs"""
    _instance = None
    
    def __new__(cls, log_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._init_logger(log_file)
        return cls._instance
    
    def _init_logger(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('PlantDisease')
        self.logger.setLevel(logging.INFO)
        
        # Supprimer les handlers existants pour éviter les doublons
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Format des logs
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Handler console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Fichier de log si spécifié
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = True):
        self.logger.error(message, exc_info=exc_info)
    
    def progress(self, iterable, desc: str = "", total: Optional[int] = None):
        return tqdm(iterable, desc=desc, total=total, ncols=100)

def ensure_dir(dir_path: str) -> Path:
    """Crée un dossier s'il n'existe pas"""
    dir_path = Path(dir_path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    except Exception as e:
        raise RuntimeError(f"Impossible de créer le dossier {dir_path}: {str(e)}")

def clear_memory():
    """Libère la mémoire GPU/CPU"""
    tf.keras.backend.clear_session()
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_allocations(gpu)

def copy_files(files: List[str], src_dir: str, dest_dir: str, class_name: str) -> None:
    """Copie des fichiers vers un dossier de destination"""
    dest_path = Path(dest_dir) / class_name
    ensure_dir(dest_path)
    
    for f in files:
        src = Path(src_dir) / f
        if src.exists():
            shutil.copy2(src, dest_path / f)
