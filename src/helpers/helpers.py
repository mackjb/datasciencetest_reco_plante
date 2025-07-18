"""
Module global pour obtenir la racine du projet via pathlib et un marker (setup.py).
"""
from pathlib import Path
import cv2


def get_project_root(marker: str = "setup.py") -> Path:
    """
    Retourne le Path du répertoire racine du projet (contenant le fichier `marker`).

    Args:
        marker: nom du fichier ou dossier marqueur (par défaut "setup.py").

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

def compute_hu_features(image_path: str) -> dict:
    """
    Calcule les 7 moments invariants de Hu en une seule passe.
    :param image_path: Chemin vers le fichier image.
    :return: Dictionnaire avec clés 'phi1_distingue_large_vs_etroit', 'phi2_distinction_elongation_forme', 
             'phi3_asymetrie_maladie', 'phi4_symetrie_diagonale_forme', 'phi5_concavite_extremites', 
             'phi6_decalage_torsion_maladie', 'phi7_asymetrie_complexe'.
    """
    # Chargement de l'image en niveaux de gris
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")
    # Calcul des moments normalisés centraux
    moments = cv2.moments(image)
    # Calcul des 7 moments de Hu
    hu = cv2.HuMoments(moments).flatten()
    return {
        'phi1_distingue_large_vs_etroit': float(hu[0]),
        'phi2_distinction_elongation_forme': float(hu[1]),
        'phi3_asymetrie_maladie': float(hu[2]),
        'phi4_symetrie_diagonale_forme': float(hu[3]),
        'phi5_concavite_extremites': float(hu[4]),
        'phi6_decalage_torsion_maladie': float(hu[5]),
        'phi7_asymetrie_complexe': float(hu[6])
    }
