#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Télécharge et prépare un nouveau dataset Kaggle dans dataset/New_Plant_Disease/.

Exemples d'usage:

1) Télécharger et déplacer vers data/
   python dataset/New_Plant_Disease/download_new_plant_disease.py --slug OWNER/DATASET_SLUG

2) Télécharger, déplacer vers data/ et créer un sous-échantillon (p.ex. 5 images par classe)
   python dataset/New_Plant_Disease/download_new_plant_disease.py --slug OWNER/DATASET_SLUG \
       --max-files-per-class 5 --subset-out data_subset

Notes:
- Nécessite kagglehub: pip install kagglehub
- Authentification Kaggle: kagglehub peut utiliser les identifiants Kaggle si configurés.
  Si vous utilisez la CLI Kaggle, placez votre kaggle.json dans ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import kagglehub

# Rendre le module helpers disponible via src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
from helpers import PROJECT_ROOT  # type: ignore


def move_dataset_if_exists(src: Path, dst: Path) -> None:
    """Déplace le dossier téléchargé `src` vers `dst`, en créant les parents si besoin."""
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"[i] Le dossier cible existe déjà, aucun déplacement: {dst}")
            return
        shutil.move(str(src), str(dst))
        print(f"✅ Déplacé :\n  {src}\n→ {dst}")
    else:
        print(f"⚠️  Répertoire source introuvable : {src}")


def duplicate_dataset_limited(src_dir: Path, dst_dir: Path, max_files_per_class: int = 5) -> None:
    """
    Copie la structure par classes en limitant le nombre d'images par classe.
    Reconnaît les extensions .jpg/.jpeg/.png (insensible à la casse).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted([d for d in src_dir.iterdir() if d.is_dir()]):
        target_class_dir = dst_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        # Lister les images
        imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        imgs = sorted(imgs)[:max_files_per_class]
        for img in imgs:
            shutil.copy2(str(img), str(target_class_dir / img.name))
    print(f"✅ Sous-échantillon créé dans {dst_dir} (max {max_files_per_class} images par classe)")


def guess_class_root(downloaded_path: Path) -> Path:
    """
    Tente de deviner le répertoire contenant les dossiers de classes.
    Si downloaded_path contient un seul sous-dossier, retourne ce sous-dossier, sinon downloaded_path.
    """
    try:
        subdirs = [d for d in downloaded_path.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
    except Exception:
        pass
    return downloaded_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharger un dataset Kaggle dans dataset/New_Plant_Disease/")
    parser.add_argument("--slug", required=True, help="Slug Kaggle du dataset (ex: owner/dataset-name)")
    parser.add_argument("--max-files-per-class", type=int, default=None,
                        help="Si défini, crée un sous-échantillon avec au plus N images par classe")
    parser.add_argument("--subset-out", type=str, default="data_subset",
                        help="Nom du dossier de sortie pour le sous-échantillon (par défaut: data_subset)")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    base_dir = project_root / "dataset" / "New_Plant_Disease"
    data_dir = base_dir / "data"

    if data_dir.exists():
        print(f"⚠️  Le dataset semble déjà présent à : {data_dir}")
    else:
        print(f"[1/3] Téléchargement Kaggle: {args.slug} ...")
        download_path = Path(kagglehub.dataset_download(args.slug))
        print(f"    → Chemin téléchargé: {download_path}")

        print("[2/3] Déplacement vers data/ ...")
        # Certains datasets s'extraient dans un sous-dossier unique, on le gère
        candidate = guess_class_root(download_path)
        move_dataset_if_exists(candidate, data_dir)

    # Optionnel: créer un sous-échantillon pour prototypage rapide
    if args.max_files_per_class is not None and args.max_files_per_class > 0:
        src_root = data_dir
        # Si data/ contient un sous-dossier unique, plonger dedans pour trouver les classes
        src_root = guess_class_root(src_root)
        subset_dir = base_dir / args.subset_out
        print(f"[3/3] Création d'un sous-échantillon: max {args.max_files_per_class}/classe → {subset_dir}")
        duplicate_dataset_limited(src_root, subset_dir, max_files_per_class=args.max_files_per_class)

    print("Terminé.")


if __name__ == "__main__":
    main()
