"""
Module global pour obtenir la racine du projet via pathlib et un marker (setup.py).
"""
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from scipy.stats import entropy
from PIL import Image, ImageStat
import math


def _to_grayscale_array(image_input):
    """Convertit chemin/ndarray/PIL.Image en ndarray grayscale"""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_input, np.ndarray):
        img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) if image_input.ndim==3 else image_input
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("L"))
    else:
        raise TypeError(f"Type non supporté: {type(image_input)}")
    if img is None: raise FileNotFoundError(f"Impossible de charger: {image_input}")
    return img


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

def compute_hog_features(image_input) -> dict:
    """
    Calcule les caractéristiques HOG agrégées (moyenne et écart-type).
    :param image_path: Chemin vers l'image.
    :return: dict avec clés 'hog_moyenne_contours_forme', 'hog_ecarttype_texture'.
    """
    image = _to_grayscale_array(image_input)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_input}")
    hog_vec = hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
    return {
        'hog_moyenne_contours_forme': float(np.mean(hog_vec)),
        'hog_ecarttype_texture': float(np.std(hog_vec))
    }

def compute_hu_features(image_input) -> dict:
    """
    Calcule les 7 moments invariants de Hu en une seule passe.
    :param image_path: Chemin vers le fichier image.
    :return: Dictionnaire avec clés 'phi1_distingue_large_vs_etroit', 'phi2_distinction_elongation_forme', 
             'phi3_asymetrie_maladie', 'phi4_symetrie_diagonale_forme', 'phi5_concavite_extremites', 
             'phi6_decalage_torsion_maladie', 'phi7_asymetrie_complexe'.
    """
    image = _to_grayscale_array(image_input)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_input}")
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


def compute_fourier_energy(image_input) -> dict:
    """
    Calcule l'énergie spectrale basse, moyenne et haute fréquences via FFT.
    :param image_path: Chemin vers le fichier image.
    :return: dict avec clés 'energie_basse_forme_feuille', 'energie_moyenne_texture_veines', 'energie_haute_details_maladie'.
    """
    image = _to_grayscale_array(image_input)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_input}")
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

def compute_pixel_ratio_and_segments(image_input) -> dict:
    """
    Calcule le ratio de pixels de feuille (foreground) et le nombre de segments.
    :param image_path: Chemin vers l'image.
    :return: dict avec clés 'pixel_ratio', 'leaf_segments'.
    """
    image = _to_grayscale_array(image_input)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_input}")
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


def is_image_valid(image_path: str) -> bool:
    """Vérifie que l'image n'est pas corrompue."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Vérifie que l'image peut être lue correctement
        return True
    except Exception:
        return False


def is_black_image(image_path, threshold=10) -> bool:
    """
    Returns True if the image is mostly black based on the mean grayscale value.
    """
    try:
        with Image.open(image_path) as img:
            stat = ImageStat.Stat(img.convert('L'))
            return stat.mean[0] < threshold
    except Exception:
        return False


def _to_rgb_array(image_input):
    """Convertit chemin/ndarray/PIL.Image en ndarray RGB"""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None: 
            raise FileNotFoundError(f"Impossible de charger: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.ndim==3 else np.stack([image_input]*3, axis=2)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    else:
        raise TypeError(f"Type non supporté: {type(image_input)}")
    return img


def compute_color_statistics(image_input) -> dict:
    """
    Calcule les statistiques de couleur en RGB et HSV.
    
    Args:
        image_input: Image PIL ou chemin vers une image
    
    Returns:
        dict: Statistiques de couleur (moyennes et écarts-types)
    """
    # Conversion en RGB
    rgb_img = _to_rgb_array(image_input)
    
    # Calcul des statistiques RGB
    r_channel = rgb_img[:, :, 0]
    g_channel = rgb_img[:, :, 1]
    b_channel = rgb_img[:, :, 2]
    
    mean_r = float(np.mean(r_channel))
    mean_g = float(np.mean(g_channel))
    mean_b = float(np.mean(b_channel))
    
    std_r = float(np.std(r_channel))
    std_g = float(np.std(g_channel))
    std_b = float(np.std(b_channel))
    
    # Conversion en HSV pour les statistiques HSV
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h_channel = hsv_img[:, :, 0]
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]
    
    mean_h = float(np.mean(h_channel))
    mean_s = float(np.mean(s_channel))
    mean_v = float(np.mean(v_channel))
    
    return {
        'mean_R': mean_r,
        'mean_G': mean_g,
        'mean_B': mean_b,
        'std_R': std_r,
        'std_G': std_g,
        'std_B': std_b,
        'mean_H': mean_h,
        'mean_S': mean_s,
        'mean_V': mean_v
    }


def compute_texture_features(image_input) -> dict:
    """
    Calcule les caractéristiques de texture basées sur la matrice GLCM (Gray-Level Co-occurrence Matrix).
    
    Args:
        image_input: Image PIL ou chemin vers une image
    
    Returns:
        dict: Caractéristiques de texture (contraste, énergie, homogénéité, etc.)
    """
    # Conversion en niveau de gris
    gray_img = _to_grayscale_array(image_input)
    
    # Normaliser l'image pour réduire le nombre de niveaux de gris (accélère le calcul)
    gray_img = (gray_img / 16).astype(np.uint8)  # Réduire à 16 niveaux de gris
    
    # Calculer GLCM pour différentes orientations (0°, 45°, 90°, 135°)
    distances = [1]  # Distance entre pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
    
    try:
        glcm = graycomatrix(
            gray_img, 
            distances=distances, 
            angles=angles, 
            symmetric=True, 
            normed=True
        )
        
        # Extraire les propriétés GLCM
        contrast = float(np.mean(graycoprops(glcm, 'contrast')[0]))
        energy = float(np.mean(graycoprops(glcm, 'energy')[0]))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')[0]))
        dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')[0]))
        correlation = float(np.mean(graycoprops(glcm, 'correlation')[0]))
        
        return {
            'contrast': contrast,
            'energy': energy,
            'homogeneity': homogeneity,
            'dissimilarite': dissimilarity,
            'correlation': correlation
        }
    except Exception as e:
        print(f"Erreur lors du calcul des caractéristiques de texture: {e}")
        # Retourner des valeurs par défaut en cas d'erreur
        return {
            'contrast': 0.0,
            'energy': 0.0,
            'homogeneity': 0.0,
            'dissimilarite': 0.0,
            'correlation': 0.0
        }


def compute_sharpness_and_contours(image_input) -> dict:
    """
    Calcule la netteté de l'image et les caractéristiques des contours.
    
    Args:
        image_input: Image PIL ou chemin vers une image
    
    Returns:
        dict: Caractéristiques de netteté et contours
    """
    # Conversion en niveau de gris
    gray_img = _to_grayscale_array(image_input)
    
    # Calcul de la netteté (variance du Laplacien)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    sharpness = float(np.var(laplacian))
    
    # Détection des contours avec Canny
    edges = cv2.Canny(gray_img, 50, 150)  # Seuils bas et haut pour Canny
    contour_pixels = np.sum(edges > 0)
    contour_density = float(contour_pixels / (gray_img.shape[0] * gray_img.shape[1]))
    
    return {
        'nettete': sharpness,
        'contour_density': contour_density
    }


def compute_additional_fft_features(image_input) -> dict:
    """
    Calcule des caractéristiques FFT supplémentaires (entropie, etc.)
    
    Args:
        image_input: Image PIL ou chemin vers une image
    
    Returns:
        dict: Caractéristiques FFT supplémentaires
    """
    # Conversion en niveau de gris
    gray_img = _to_grayscale_array(image_input)
    
    # Transformée de Fourier
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Normalisation du spectre pour calculer l'entropie
    magnitude_spectrum_norm = magnitude_spectrum / np.sum(magnitude_spectrum)
    magnitude_spectrum_flat = magnitude_spectrum_norm.flatten()
    magnitude_spectrum_flat = magnitude_spectrum_flat[magnitude_spectrum_flat > 0]  # Éviter log(0)
    
    # Calcul de l'entropie
    try:
        fft_entropy_val = float(entropy(magnitude_spectrum_flat))
    except Exception:
        fft_entropy_val = 0.0
    
    return {
        'fft_entropy': fft_entropy_val
    }


def compute_additional_hog_features(image_input) -> dict:
    """
    Calcule des caractéristiques HOG supplémentaires (entropie)
    
    Args:
        image_input: Image PIL ou chemin vers une image
    
    Returns:
        dict: Caractéristiques HOG supplémentaires
    """
    # Conversion en niveau de gris
    gray_img = _to_grayscale_array(image_input)
    
    # Calculer les caractéristiques HOG
    try:
        hog_vec = hog(gray_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), 
                      block_norm='L2-Hys', feature_vector=True)
        
        # Normalisation pour calculer l'entropie
        hog_norm = hog_vec / np.sum(hog_vec) if np.sum(hog_vec) > 0 else hog_vec
        hog_norm = hog_norm[hog_norm > 0]  # Éviter log(0)
        
        # Calcul de l'entropie
        hog_entropy_val = float(entropy(hog_norm)) if len(hog_norm) > 0 else 0.0
        
        return {
            'hog_entropy': hog_entropy_val
        }
    except Exception as e:
        print(f"Erreur lors du calcul des caractéristiques HOG supplémentaires: {e}")
        return {
            'hog_entropy': 0.0
        }


def compute_shape_features(image_input) -> dict:
    """
    Calcule les descripteurs de forme (area, perimeter, circularity, etc.)
    
    Args:
        image_input: Image PIL ou chemin vers une image
    
    Returns:
        dict: Descripteurs de forme de l'objet principal de l'image
    """
    try:
        # Conversion en niveau de gris
        gray_img = _to_grayscale_array(image_input)
        
        # Binarisation pour segmentation
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Recherche des contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si aucun contour n'est trouvé
        if not contours:
            return {
                'area': 0.0, 'perimeter': 0.0, 'circularity': 0.0,
                'solidity': 0.0, 'extent': 0.0, 'eccentricity': 0.0,
                'major_axis_length': 0.0, 'minor_axis_length': 0.0,
                'compactness': 0.0, 'fractal_dimension': 0.0
            }
        
        # Trouver le plus grand contour (supposant que c'est l'objet principal)
        c = max(contours, key=cv2.contourArea)
        
        # Calcul des descripteurs de forme
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        
        # Éviter division par zéro
        if perimeter == 0:
            perimeter = 1.0
        
        # Circularity = 4*pi*area/perimeter^2 (1 pour cercle parfait, < 1 pour formes irrégulières)
        circularity = float((4 * np.pi * area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
        
        # Enveloppe convexe et sa surface
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        
        # Solidity = area/hull_area (1 pour objet convexe, < 1 sinon)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        
        # Rectangle englobant et son aire
        x, y, w, h = cv2.boundingRect(c)
        rect_area = float(w * h)
        
        # Extent = area/rect_area (1 pour rectangle, < 1 sinon)
        extent = float(area / rect_area) if rect_area > 0 else 0.0
        
        # Ellipse englobante
        if len(c) >= 5:  # Besoin d'au moins 5 points pour ellipse
            (x, y), (width, height), angle = cv2.fitEllipse(c)
            major_axis = max(width, height) / 2
            minor_axis = min(width, height) / 2
            
            # Eccentricité de l'ellipse
            eccentricity = float(np.sqrt(1 - ((minor_axis * minor_axis) / (major_axis * major_axis)))) if major_axis > 0 else 0.0
            
            major_axis_length = float(major_axis)
            minor_axis_length = float(minor_axis)
        else:
            # Valeurs par défaut si pas assez de points
            eccentricity = 0.0
            major_axis_length = 0.0
            minor_axis_length = 0.0
        
        # Compactness = sqrt(4*area/pi) / major_axis_length
        # Mesure à quel point l'objet est proche d'un cercle par rapport à son axe principal
        compactness = float(np.sqrt(4 * area / np.pi) / major_axis_length) if major_axis_length > 0 else 0.0
        
        # Dimension fractale (estimation simplifiée basée sur le périmètre et l'aire)
        # Plus la dimension est élevée, plus le contour est complexe
        # D = 2 * log(P) / log(A) où P est le périmètre et A l'aire
        if area > 0:
            try:
                fractal_dimension = float(2 * np.log(perimeter) / np.log(area))
            except:
                fractal_dimension = 0.0
        else:
            fractal_dimension = 0.0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'solidity': solidity,
            'extent': extent,
            'eccentricity': eccentricity,
            'major_axis_length': major_axis_length,
            'minor_axis_length': minor_axis_length,
            'compactness': compactness,
            'fractal_dimension': fractal_dimension
        }
        
    except Exception as e:
        print(f"Erreur lors du calcul des descripteurs de forme: {e}")
        return {
            'area': 0.0, 'perimeter': 0.0, 'circularity': 0.0,
            'solidity': 0.0, 'extent': 0.0, 'eccentricity': 0.0,
            'major_axis_length': 0.0, 'minor_axis_length': 0.0,
            'compactness': 0.0, 'fractal_dimension': 0.0
        }
