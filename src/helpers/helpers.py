"""
Module global pour obtenir la racine du projet via pathlib et un marker (setup.py).
"""
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog


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

def compute_hog_features(image_path: str) -> dict:
    """
    Calcule les caractéristiques HOG agrégées (moyenne et écart-type).
    :param image_path: Chemin vers l'image.
    :return: dict avec clés 'hog_moyenne_contours_forme', 'hog_ecarttype_texture'.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")
    hog_vec = hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
    return {
        'hog_moyenne_contours_forme': float(np.mean(hog_vec)),
        'hog_ecarttype_texture': float(np.std(hog_vec))
    }

def compute_hu_features(image_path: str) -> dict:
    """
    Calcule les 7 moments invariants de Hu en une seule passe.
    :param image_path: Chemin vers le fichier image.
    :return: Dictionnaire avec clés 'phi1_distingue_large_vs_etroit', 'phi2_distinction_elongation_forme', 
             'phi3_asymetrie_maladie', 'phi4_symetrie_diagonale_forme', 'phi5_concavite_extremites', 
             'phi6_decalage_torsion_maladie', 'phi7_asymetrie_complexe'.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")
    moments = cv2.moments(image)
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


def compute_fourier_energy(image_path: str) -> dict:
    """
    Calcule l'énergie spectrale basse, moyenne et haute fréquences via FFT.
    :param image_path: Chemin vers le fichier image.
    :return: dict avec clés 'energie_basse_forme_feuille', 'energie_moyenne_texture_veines', 'energie_haute_details_maladie'.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")
    # Transformée de Fourier
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift)**2
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # Matrice de distances radiales
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow)**2 + (x - ccol)**2)
    r_max = distance.max()
    # Définition des bandes de fréquences
    low_mask = distance <= r_max / 4
    med_mask = (distance > r_max / 4) & (distance <= r_max / 2)
    high_mask = distance > r_max / 2
    total_energy = power.sum()
    if total_energy == 0:
        return {
            'energie_basse_forme_feuille': 0.0,
            'energie_moyenne_texture_veines': 0.0,
            'energie_haute_details_maladie': 0.0
        }
    # Énergies normalisées
    lb = power[low_mask].sum() / total_energy
    mb = power[med_mask].sum() / total_energy
    hb = power[high_mask].sum() / total_energy
    return {
        'energie_basse_forme_feuille': float(lb),
        'energie_moyenne_texture_veines': float(mb),
        'energie_haute_details_maladie': float(hb)
    }

def compute_pixel_ratio_and_segments(image_path: str) -> dict:
    """
    Calcule le ratio de pixels de feuille (foreground) et le nombre de segments.
    :param image_path: Chemin vers l'image.
    :return: dict avec clés 'pixel_ratio', 'leaf_segments'.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")
    # Foreground mask: pixels > 0 (non-noir)
    mask = image > 0
    total = mask.size
    leaf_pixels = int(mask.sum())
    pixel_ratio = leaf_pixels / total if total else 0.0
    # Compter les composants connectés (0: background)
    mask_uint8 = mask.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_uint8)
    leaf_segments = num_labels - 1
    return {
        'pixel_ratio': float(pixel_ratio),
        'leaf_segments': int(leaf_segments)
    }
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
