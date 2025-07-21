"""
Module global pour obtenir la racine du projet Git via pathlib et un marker (.gitignore).
"""
from pathlib import Path


def get_project_root(marker: str = ".gitignore") -> Path:
    """
    Retourne le Path du répertoire racine du projet (contenant le fichier `marker`).

    Args:
        marker: nom du fichier ou dossier marqueur (par défaut ".gitignore").

    Returns:
        Path vers la racine du projet.

    Raises:
        FileNotFoundError: si aucun parent ne contient `marker`.
    """
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Impossible de trouver la racine du projet (marker={marker})")

# Instance globale accessible partout
PROJECT_ROOT: Path = get_project_root()
