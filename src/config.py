from pathlib import Path
import yaml
from typing import Any, Dict, Optional, Union, TypeVar
import os
import logging
from functools import wraps

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

def log_config_operation(func):
    """Décorateur pour logger les opérations de configuration."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Appel de {func.__name__} avec args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Erreur dans {func.__name__}: {str(e)}")
            raise
    return wrapper

class Config:
    """Classe singleton pour la gestion de la configuration de l'application.
    
    Charge la configuration depuis le fichier YAML et fournit des méthodes pour y accéder.
    """
    _instance = None
    _config = {}
    _repo_root = Path(__file__).resolve().parent.parent
    
    # Chemins obligatoires qui doivent exister
    REQUIRED_PATHS = [
        'data.raw',
        'data.processed',
        'data.external',
        'output.logs',
        'output.models',
        'output.results',
        'output.reports'
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    @log_config_operation
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier YAML."""
        try:
            from configs import DEFAULT_CONFIG
            cfg_path = DEFAULT_CONFIG
            
            logger.info(f"Chargement de la configuration depuis {cfg_path}")
            
            if not cfg_path.exists():
                error_msg = f"Fichier de configuration introuvable : {cfg_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            with open(cfg_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
                
            if not self._config:
                logger.warning("Le fichier de configuration est vide")
            
            # Création des répertoires nécessaires
            self._create_directories()
            
            # Validation des chemins requis
            self._validate_required_paths()
            
            logger.info("Configuration chargée avec succès")
            
        except yaml.YAMLError as e:
            error_msg = f"Erreur de syntaxe dans le fichier de configuration : {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Erreur lors du chargement de la configuration : {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @log_config_operation
    def _create_directories(self) -> None:
        """Crée les répertoires de données et de sortie s'ils n'existent pas."""
        dirs_to_create = []
        
        # Ajouter les répertoires de base
        base_dirs = [
            self._repo_root / 'data',
            self._repo_root / 'logs',
            self._repo_root / 'models',
            self._repo_root / 'results'
        ]
        dirs_to_create.extend([d for d in base_dirs if d and not d.exists()])
        
        # Ajouter les répertoires de configuration
        for path_key in self.REQUIRED_PATHS:
            dir_path = self.get(path_key, path_only=True, default=None)
            if dir_path and not dir_path.exists():
                dirs_to_create.append(dir_path)
        
        # Créer les répertoires manquants
        for dir_path in dirs_to_create:
            try:
                dir_path = Path(dir_path)
                if not dir_path.exists():
                    logger.info(f"Création du répertoire : {dir_path}")
                    dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Impossible de créer le répertoire {dir_path}: {e}")
                raise
    
    def _validate_required_paths(self) -> None:
        """Valide que tous les chemins requis existent ou peuvent être créés."""
        missing_paths = []
        
        for path_key in self.REQUIRED_PATHS:
            path = self.get(path_key, path_only=True, default=None)
            if not path:
                missing_paths.append(f"{path_key} (non défini)")
            elif not path.exists() and not path.parent.exists():
                # Si ni le chemin ni son parent n'existent, on ne peut pas le créer
                missing_paths.append(f"{path_key} ({path} - répertoire parent manquant)")
        
        if missing_paths:
            error_msg = (
                "Chemins de configuration manquants ou invalides :\n"
                f"{chr(10).join(f'- {p}' for p in missing_paths)}\n"
                "et que les répertoires parents existent."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    @log_config_operation
    def get(self, key: str, default: Any = None, path_only: bool = False) -> Any:
        """
        Récupère une valeur de configuration par clé.
        
        Args:
            key: Clé de configuration au format 'section.subsection.key'
            default: Valeur par défaut si la clé n'existe pas
            path_only: Si True, convertit la valeur en objet Path
            
        Returns:
            La valeur de configuration ou la valeur par défaut
            
        Raises:
            KeyError: Si la clé n'existe pas et qu'aucune valeur par défaut n'est fournie
        """
        try:
            # Parcours des clés imbriquées
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if not isinstance(value, dict) or k not in value:
                    if default is not None:
                        value = default
                        break
                    raise KeyError(f"Clé de configuration non trouvée : {key}")
                value = value[k]
            
            # Conversion en Path si demandé et que c'est une chaîne
            if path_only and value is not None:
                if isinstance(value, str):
                    # Si c'est un chemin relatif, on le rend relatif à la racine du projet
                    path = Path(value)
                    return path if path.is_absolute() else (self._repo_root / path).resolve()
                elif isinstance(value, Path):
                    return value
                else:
                    logger.warning(f"Impossible de convertir en Path: {value}")
                    return None
                    
            # Si c'est un chemin et que path_only est True, retourne un objet Path
            if path_only and isinstance(value, str):
                # Vérifie si c'est un chemin relatif au projet
                if not os.path.isabs(value):
                    return self._repo_root / value
                return Path(value)
                
            return value
            
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """Retourne une copie du dictionnaire de configuration."""
        return self._config.copy()

    def __str__(self) -> str:
        """Représentation lisible de la configuration."""
        import json
        from pathlib import Path
        
        def json_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Type non sérialisable : {type(obj)}")
        
        return json.dumps(
            self._config,
            indent=2,
            default=json_serializable,
            ensure_ascii=False
        )

# Instance globale pour une utilisation facile
config = Config()

# Vérification de la configuration au chargement du module
try:
    # Vérifie que les chemins requis sont accessibles
    for path_key in config.REQUIRED_PATHS:
        config.get(path_key, path_only=True)
    logger.info("Configuration validée avec succès")
except Exception as e:
    logger.error(f"Erreur de validation de la configuration : {e}")
    raise
