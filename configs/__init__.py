"""
Package de configuration du projet.

Ce package contient les fichiers de configuration au format YAML utilisés par l'application.

Structure de configuration :
- `default.yaml` : Configuration principale avec les paramètres par défaut
- Les configurations sont chargées via `src.config.Config`

Exemple d'utilisation :
    >>> from src.config import config
    >>> 
    >>> # Accéder à une valeur de configuration
    >>> data_path = config.get('data.raw')
    >>> 
    # Utiliser un chemin de configuration
    >>> from pathlib import Path
    >>> model_dir = Path(config.get('models.yolov8.output_dir'))

Note :
- Tous les chemins dans les fichiers de configuration sont relatifs à la racine du projet.
- Les chemins sont automatiquement convertis en objets Path lors du chargement.
"""

from pathlib import Path

# Chemin absolu vers le répertoire de configuration
CONFIG_DIR = Path(__file__).parent.absolute()

# Fichiers de configuration disponibles
DEFAULT_CONFIG = CONFIG_DIR / 'default.yaml'

__all__ = ['DEFAULT_CONFIG']
