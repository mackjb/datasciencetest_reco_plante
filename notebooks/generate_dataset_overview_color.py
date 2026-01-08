from pathlib import Path
import random
import csv
from PIL import Image
import matplotlib.pyplot as plt

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossiers PlantVillage
COLOR_ROOT = PROJECT_ROOT / "dataset/plantvillage/data/plantvillage dataset/color"
SEG_ROOT = PROJECT_ROOT / "dataset/plantvillage/data/plantvillage dataset/segmented"

# Fichiers de sortie
OUT_COLOR = PROJECT_ROOT / "Streamlit/assets/dataset_overview_color_select.png"
OUT_SEG = PROJECT_ROOT / "Streamlit/assets/dataset_overview_segmented_select.png"

print("COLOR_ROOT:", COLOR_ROOT)
print("SEG_ROOT:", SEG_ROOT)
assert COLOR_ROOT.exists(), "Le dossier des images color n'existe pas."
assert SEG_ROOT.exists(), "Le dossier des images segmented n'existe pas."

# Espèces à illustrer : (titre, préfixes des dossiers)
SPECIES_CONFIG = [
    ("Pomme (Apple)", ["Apple___"]),
    ("Maïs (Corn)", ["Corn_(maize)___"]),
    ("Myrtille (Blueberry)", ["Blueberry___"]),
    ("Cerise (Cherry)", ["Cherry_(including_sour)___"]),
    ("Raisin (Grape)", ["Grape___"]),
    ("Pomme de Terre (Potato)", ["Potato___"]),
    ("Soja (Soybean)", ["Soybean___"]),
    ("Fraise (Strawberry)", ["Strawberry___"]),
]

# nb de colonnes = nb d'espèces
n_cols = len(SPECIES_CONFIG)
# nb de lignes = nb d'images par espèce
n_rows = 2

species_per_row = 4

random.seed(42)


def pick_color_for_species(prefixes, max_images):
    """Sélectionne des images COLOR pour une espèce donnée."""
    imgs = []
    for d in COLOR_ROOT.iterdir():
        if not d.is_dir():
            continue
        if any(d.name.startswith(p) for p in prefixes):
            files = list(d.glob("*.jpg")) + list(d.glob("*.JPG")) + list(d.glob("*.png"))
            imgs.extend(files)
    random.shuffle(imgs)
    return imgs[:max_images]


def pick_color_with_segmented(prefixes, max_images):
    """Sélectionne des images COLOR qui ont un équivalent SEGMENTED (même base de nom)."""
    seg_idx = _segmented_index_for_prefixes(prefixes)

    imgs = []
    for d in COLOR_ROOT.iterdir():
        if not d.is_dir():
            continue
        if any(d.name.startswith(p) for p in prefixes):
            files = list(d.glob("*.jpg")) + list(d.glob("*.JPG")) + list(d.glob("*.png"))
            imgs.extend([p for p in files if p.stem in seg_idx])

    random.shuffle(imgs)
    return imgs[:max_images]


def _segmented_index_for_prefixes(prefixes):
    """Indexe les fichiers segmented pour une espèce, pour retrouver un équivalent par nom."""
    index = {}
    for d in SEG_ROOT.iterdir():
        if not d.is_dir():
            continue
        if any(d.name.startswith(p) for p in prefixes):
            files = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.JPG"))
            for p in files:
                stem = p.stem
                base = stem[:-13] if stem.endswith("_final_masked") else stem
                index.setdefault(base, []).append(p)

    for base in index:
        index[base] = sorted(index[base])
    return index


def pick_segmented_matching_color(prefixes, color_imgs):
    """Pour chaque image COLOR sélectionnée, récupère l'image SEGMENTED correspondante (même base de nom)."""
    idx = _segmented_index_for_prefixes(prefixes)

    seg_imgs = []
    for c in color_imgs:
        base = c.stem
        candidates = idx.get(base)
        if not candidates:
            continue

        # Priorité à *_final_masked.png si présent
        final_masked = [p for p in candidates if p.stem == f"{base}_final_masked"]
        seg_imgs.append(final_masked[0] if final_masked else candidates[0])

    return seg_imgs


# Préparer les listes d'images par espèce
species_color = []
species_seg = []

for title, prefixes in SPECIES_CONFIG:
    c_imgs = pick_color_with_segmented(prefixes, n_rows)
    s_imgs = pick_segmented_matching_color(prefixes, c_imgs)
    print(f"{title}: {len(c_imgs)} color, {len(s_imgs)} segmented")
    species_color.append((title, c_imgs))
    species_seg.append((title, s_imgs))


def write_manifest(species_images, manifest_path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["output", "species_title", "row", "image_path"])
        for title, imgs in species_images:
            for i, p in enumerate(imgs):
                writer.writerow([manifest_path.stem, title, i, str(p)])


def build_grid(species_images, output_path, title_suffix=""):
    """Construit une grille (n_rows x n_cols) à partir d'une liste [(titre, [images...]), ...]."""
    total_species = len(species_images)
    n_species_rows = (total_species + species_per_row - 1) // species_per_row

    grid_cols = min(species_per_row, total_species)
    grid_rows = n_rows * n_species_rows

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3 * grid_cols, 3 * grid_rows))

    # Normalisation de la forme de axes
    if grid_rows == 1:
        axes = axes[None, :]
    if grid_cols == 1:
        axes = axes[:, None]

    # Mettre tout à "off" par défaut
    for ax in axes.flatten():
        ax.axis("off")

    for s_idx, (title, imgs) in enumerate(species_images):
        block_row = s_idx // species_per_row
        block_col = s_idx % species_per_row

        # Le titre sur la 1ère ligne du bloc
        axes[block_row * n_rows, block_col].set_title(title + title_suffix, fontsize=10)

        for r in range(n_rows):
            ax = axes[block_row * n_rows + r, block_col]
            if r < len(imgs):
                img = Image.open(imgs[r])
                ax.imshow(img)
            ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Image sauvegardée dans:", output_path)


# Générer les deux grilles
build_grid(species_color, OUT_COLOR)
build_grid(species_seg, OUT_SEG)

write_manifest(
    species_color,
    PROJECT_ROOT / "Streamlit/assets/dataset_overview_color_select_manifest.csv",
)
write_manifest(
    species_seg,
    PROJECT_ROOT / "Streamlit/assets/dataset_overview_segmented_select_manifest.csv",
)