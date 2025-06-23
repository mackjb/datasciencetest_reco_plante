# sam_extract_mask.py
# 
# Utilise le modèle Segment Anything de Meta pour projeter 4 points aux extrémités
# de l'image, extraire le mask correspondant et générer une nouvelle image nommée "resultat.png".

import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

# CHEMIN_VERS_CHECKPOINT: remplacez par le chemin vers le fichier .pth du modèle SAM
SAM_CHECKPOINT = "chemin/vers/sam_vit_b_01ec64.pth"
# Choix du modèle: "vit_b", "vit_l", "vit_h"
MODEL_TYPE = "vit_b"
# Chemin de l'image d'entrée
INPUT_IMAGE_PATH = "input.jpg"
# Nom de l'image de sortie
OUTPUT_IMAGE_PATH = "resultat.png"


def main():
    # Charger l'image
    image_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    image = np.array(image_pil)
    height, width, _ = image.shape

    # Initialiser le modèle SAM
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # Définir les 4 points aux coins de l'image
    # Format: [[x, y], ...]
    corners = np.array([
        [0, 0],            # coin supérieur gauche
        [width - 1, 0],    # coin supérieur droit
        [width - 1, height - 1],  # coin inférieur droit
        [0, height - 1]    # coin inférieur gauche
    ])
    # Labels: 1 pour foreground
    labels = np.ones(corners.shape[0], dtype=int)

    # Prédire le mask pour ces points
    masks, scores, logits = predictor.predict(
        point_coords=corners,
        point_labels=labels,
        multimask_output=False
    )

    # On récupère le premier (et unique) mask
    mask = masks[0]

    # Appliquer le mask sur l'image d'origine
    # Les pixels hors mask deviennent noirs
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # Enregistrer le résultat
    result_pil = Image.fromarray(masked_image)
    result_pil.save(OUTPUT_IMAGE_PATH)
    print(f"Image enregistrée sous: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
