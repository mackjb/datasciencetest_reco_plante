#!/usr/bin/env python3
"""
Génère un CSV nettoyé à partir du dataset PlantVillage (segmented) avec pipeline:
- Standardisation des images en JPG 256x256 dans un dossier propre (clean_plantvillage_dataset)
- Tests de qualité appliqués APRÈS conversion/redimensionnement:
  - is_image_valid(path)
  - is_black (seuil de luminance)
- Dé-duplication basée sur le MD5 des images converties (on garde le premier MD5)

Sortie CSV: dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv
Source:     dataset/plantvillage/data/plantvillage dataset/segmented
Images out: dataset/plantvillage/data/clean_plantvillage_dataset
"""
from __future__ import annotations
import os
import sys
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict
import io
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assurer l'import des helpers via src/
THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import pandas as pd
from PIL import Image, ImageOps, ImageStat
import cv2
import numpy as np
import base64
# Importer directement depuis le module helpers.helpers
from helpers.helpers import PROJECT_ROOT, is_image_valid, is_black_image


def compute_md5(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    """Calcule le hash MD5 d'un fichier, en lisant par chunks pour éviter la surcharge mémoire."""
    try:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        print(f"[MD5] Erreur pour {file_path}: {e}")
        return None


def compute_md5_bytes(content: bytes) -> str:
    """Calcule le MD5 d'un contenu binaire en mémoire."""
    md5 = hashlib.md5()
    md5.update(content)
    return md5.hexdigest()


def _is_black_pil(img: Image.Image, threshold: int = 5) -> bool:
    """Détecte si l'image PIL est quasi noire via la moyenne en niveaux de gris."""
    try:
        stat = ImageStat.Stat(img.convert('L'))
        return stat.mean[0] < threshold
    except Exception:
        return False


# Compatibilité Pillow < 10 / >= 10 pour le filtre de redimensionnement
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS  # Pillow >= 10
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS  # Pillow < 10


def build_clean_dataset(
    src_root: Path,
    dst_images_root: Path,
    black_threshold: int = 5,
    max_workers: int | None = None,
    # paramètres composantes connexes
    cc_gray_threshold: int = 5,
    cc_min_area_ratio: float = 0.01,
    cc_min_area_pixels: int = 0,
    cc_flag_threshold: int = 3,
    purge_dst: bool = True,
) -> pd.DataFrame:
    """
    Parcourt `src_root`, convertit chaque image valide en JPG 256x256, applique un test d'image noire
    APRÈS conversion, dé-duplique via MD5 du JPG converti, et enregistre dans `dst_images_root`.
    Retourne un DataFrame des images retenues pointant vers les chemins nettoyés.
    """
    if not src_root.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {src_root}")

    # Purger la destination si demandé pour éviter de conserver d'anciens fichiers
    if purge_dst and dst_images_root.exists():
        shutil.rmtree(dst_images_root)
    dst_images_root.mkdir(parents=True, exist_ok=True)

    # Collecte des fichiers source
    allowed_ext = (".jpg", ".jpeg", ".png")
    files: List[Path] = []
    for dirpath, _dirs, fnames in os.walk(src_root):
        for fname in fnames:
            if fname.lower().endswith(allowed_ext):
                files.append(Path(dirpath) / fname)

    total_files = len(files)

    seen_md5: set[str] = set()
    lock = threading.Lock()
    filtered_relpaths: List[tuple[str, int]] = []  # (relative_path, num_large_components)

    counters = {
        "invalid": 0,
        "black": 0,
        "dups": 0,
        "errors": 0,
        "saved": 0,
        "filtered_components": 0,
    }

    records: List[Dict] = []

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)

    def _count_large_components(img: Image.Image) -> int:
        # Compter les gros éléments sur l'image 256x256 (post-conversion)
        gray = np.array(img.convert('L'))
        _, binary = cv2.threshold(gray, cc_gray_threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        total_pixels = gray.shape[0] * gray.shape[1]
        thr_pix = int(cc_min_area_ratio * total_pixels)
        if cc_min_area_pixels and cc_min_area_pixels > 0:
            thr_pix = max(thr_pix, int(cc_min_area_pixels))
        return int((areas >= thr_pix).sum())

    def _write_filtered_gallery(items: List[tuple[str, int]]):
        if not items:
            return None
        results_dir = PROJECT_ROOT / "results" / "filter_components"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_html = results_dir / "filtered_components_gallery.html"

        # Build HTML with base64 thumbnails for portability and link to source images
        cards_html = []
        for rel, n in items:
            src_path = src_root / rel
            try:
                with Image.open(src_path) as im:
                    im = ImageOps.exif_transpose(im)
                    im = im.convert("RGB")
                    im.thumbnail((256, 256), RESAMPLE_FILTER)
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
                # Relative href from gallery to source image under dataset/.../segmented
                try:
                    rel_href = os.path.relpath(src_path, results_dir)
                except Exception:
                    rel_href = str(src_path)
                rel_href = rel_href.replace(os.sep, '/')
                rel_href = rel_href.replace(' ', '%20')
                caption = f"{rel} | components: {n}"
                cards_html.append(
                    f'<a class="card" href="{rel_href}" target="_blank">'
                    f'<img src="data:image/jpeg;base64,{b64}" alt="{rel}">'
                    f'<div class="cap">{caption}</div>'
                    f'</a>'
                )
            except Exception:
                # If loading fails, still show the path
                cards_html.append(f'<div class="card"><div class="cap">{rel} (failed to load)</div></div>')

        html = f"""
<!DOCTYPE html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\" />
  <title>Filtered images (> {cc_flag_threshold} components)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    h1 {{ font-size: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ display: block; text-decoration: none; color: inherit; border: 1px solid #ddd; border-radius: 8px; padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .card img {{ width: 100%; height: auto; border-radius: 4px; display: block; }}
    .cap {{ font-size: 12px; margin-top: 6px; word-break: break-word; }}
  </style>
  </head>
  <body>
    <h1>Images filtrées (&gt; {cc_flag_threshold} gros éléments): {len(items)}</h1>
    <div class=\"grid\">
      {''.join(cards_html)}
    </div>
  </body>
</html>
"""
        out_html.write_text(html, encoding="utf-8")
        print(f"📄 Galerie créée: {out_html}")
        return out_html

    def worker(fpath: Path):
        # 1) Vérif corruption rapide
        try:
            if not is_image_valid(str(fpath)):
                return None, {"invalid": 1}
        except Exception:
            return None, {"invalid": 1}

        # 2) Ouverture et conversion -> JPG 256x256
        try:
            with Image.open(fpath) as img:
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                img = img.convert("RGB")
                img = img.resize((256, 256), RESAMPLE_FILTER)

                # 3) Test 'black' après conversion
                if _is_black_pil(img, threshold=black_threshold):
                    return None, {"black": 1}

                # Détecter l'espèce pour ignorer le filtre sur Strawberry
                try:
                    _relative_pre = fpath.relative_to(src_root)
                except Exception:
                    _relative_pre = fpath
                _parts_pre = _relative_pre.parts
                _classe_pre = _parts_pre[0] if len(_parts_pre) >= 2 else "Unknown"
                try:
                    _nom_plante_pre, _ = _classe_pre.split("___")
                except Exception:
                    _nom_plante_pre = _classe_pre
                _skip_cc = (_nom_plante_pre.lower() == "strawberry")

                # 3b) Filtre composantes connexes: exclure si > cc_flag_threshold (sauf Strawberry)
                if not _skip_cc:
                    num_large = _count_large_components(img)
                    if num_large > cc_flag_threshold:
                        # Chemin relatif pour impression
                        try:
                            relative = fpath.relative_to(src_root)
                            rel_str = str(relative)
                        except Exception:
                            rel_str = fpath.name
                        with lock:
                            filtered_relpaths.append((rel_str, num_large))
                        return None, {"filtered_components": 1}

                # 4) Encoder en JPEG (mémoire) pour MD5 et écriture disque
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95, optimize=True)
                content = buf.getvalue()
        except Exception:
            return None, {"errors": 1}

        # 5) Dé-dup par MD5 du JPG converti
        md5 = compute_md5_bytes(content)
        with lock:
            if md5 in seen_md5:
                return None, {"dups": 1}
            seen_md5.add(md5)

        # 6) Déterminer labels et chemin de sortie
        try:
            try:
                relative = fpath.relative_to(src_root)
            except ValueError:
                relative = fpath
            parts = relative.parts
            if len(parts) < 2:
                # structure inattendue
                classe = "Unknown"
            else:
                classe = parts[0]
            try:
                nom_plante, nom_maladie = classe.split("___")
            except Exception:
                nom_plante, nom_maladie = classe, "Unknown"

            est_saine = (nom_maladie.lower() == "healthy")

            out_dir = dst_images_root / classe
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(fpath).stem}_{md5[:8]}.jpg"
            out_path = out_dir / out_name

            # 7) Écriture disque du JPEG
            try:
                out_path.write_bytes(content)
            except Exception:
                return None, {"errors": 1}

            rec = {
                "nom_plante": nom_plante,
                "nom_maladie": nom_maladie,
                "Est_Saine": est_saine,
                "Image_Path": str(out_path),
                "width_img": 256,
                "height_img": 256,
                "is_black": False,
                "md5": md5,
                # Optionnel: utile pour audit
                # "num_large_components": num_large,
            }
            return rec, {"saved": 1}
        except Exception:
            return None, {"errors": 1}

    print("--- Démarrage pipeline clean (JPG 256x256) ---")
    print(f"Source:      {src_root}")
    print(f"Destination images: {dst_images_root}")
    print(f"Fichiers à traiter: {total_files}")
    print(f"max_workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, fp) for fp in files]
        for fut in as_completed(futures):
            rec, cnt = fut.result()
            if rec is not None:
                records.append(rec)
            for k, v in cnt.items():
                counters[k] += v

    print("--- Statistiques ---")
    print(f"Total fichiers image scannés: {total_files}")
    print(f"Images invalides ignorées:   {counters['invalid']}")
    print(f"Images 'presque noires':     {counters['black']}")
    print(f"Doublons MD5 ignorés:        {counters['dups']}")
    print(f"Erreurs traitement:          {counters['errors']}")
    print(f"Images retenues:             {counters['saved']}")
    print(f"Images filtrées (composants> {cc_flag_threshold}): {counters['filtered_components']}")

    if filtered_relpaths:
        print("--- Images filtrées (relative au src_root) ---")
        for p, n in filtered_relpaths:
            print(f" - {p} (num_large={n})")
        # Générer une galerie HTML dans results/
        _write_filtered_gallery(filtered_relpaths)

    return pd.DataFrame(records)


def collect_image_metadata(root_dir: Path, black_threshold: int = 5) -> pd.DataFrame:
    """
    Parcourt le répertoire `root_dir` et construit un DataFrame avec filtrage:
    - Garde uniquement images valides (PIL peut les vérifier)
    - Exclut images quasi noires (threshold niveaux de gris)
    - Dé-duplique par MD5 (garde la première occurrence)

    Colonnes retournées:
    ['nom_plante','nom_maladie','Est_Saine','Image_Path','width_img','height_img','is_black','md5']
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {root_dir}")

    data: List[Dict] = []
    seen_md5: set[str] = set()

    cnt_total = 0
    cnt_invalid = 0
    cnt_black = 0
    cnt_dups = 0

    # Parcours récursif
    for dirpath, _dirs, files in os.walk(root_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            cnt_total += 1
            fpath = Path(dirpath) / fname

            # 1) Valide
            if not is_image_valid(str(fpath)):
                cnt_invalid += 1
                continue

            # 2) Noir ?
            if is_black_image(str(fpath), threshold=black_threshold):
                cnt_black += 1
                continue

            # 3) MD5 pour dé-duplication
            md5 = compute_md5(fpath)
            if md5 is None:
                # si impossible de calculer, on passe
                continue
            if md5 in seen_md5:
                cnt_dups += 1
                continue
            seen_md5.add(md5)

            # Extraire labels depuis le dossier racine (species___disease)
            try:
                relative = fpath.relative_to(root_dir)
            except ValueError:
                relative = fpath
            parts = relative.parts
            if len(parts) < 2:
                # structure inattendue, on ignore
                continue
            dossier = parts[0]
            try:
                nom_plante, nom_maladie = dossier.split("___")
            except Exception:
                nom_plante, nom_maladie = dossier, "Unknown"
            est_saine = (nom_maladie.lower() == "healthy")

            # Dimensions
            try:
                with Image.open(fpath) as img:
                    width, height = img.size
            except Exception:
                width, height = None, None

            data.append({
                "nom_plante": nom_plante,
                "nom_maladie": nom_maladie,
                "Est_Saine": est_saine,
                "Image_Path": str(fpath),
                "width_img": width,
                "height_img": height,
                "is_black": False,
                "md5": md5,
            })

    print("--- Statistiques ---")
    print(f"Total fichiers image scannés: {cnt_total}")
    print(f"Images invalides ignorées:   {cnt_invalid}")
    print(f"Images 'presque noires':     {cnt_black}")
    print(f"Doublons MD5 ignorés:        {cnt_dups}")
    print(f"Images retenues:             {len(data)}")
    return pd.DataFrame(data)


def main() -> None:
    # Chemins source/destination en s'appuyant sur PROJECT_ROOT
    src_root = PROJECT_ROOT / "dataset" / "plantvillage" / "data" / "plantvillage dataset" / "segmented"
    dst_images_root = PROJECT_ROOT / "dataset" / "plantvillage" / "data" / "clean_plantvillage_dataset"
    dst_csv = PROJECT_ROOT / "dataset" / "plantvillage" / "csv" / "clean_data_plantvillage_segmented_all.csv"

    print(f"Source images: {src_root}")
    print(f"Images nettoyées -> {dst_images_root}")
    print(f"CSV destination: {dst_csv}")

    t0 = time.perf_counter()
    df = build_clean_dataset(src_root, dst_images_root, black_threshold=5, max_workers=None)
    t1 = time.perf_counter()

    # Création du dossier de sortie CSV
    dst_csv.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde CSV
    df.to_csv(dst_csv, index=False)
    print(f"✅ CSV écrit: {dst_csv} ({len(df)} lignes)")
    elapsed = t1 - t0
    print(f"⏱️ Temps de traitement: {elapsed:.2f} secondes (~{elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
