import os
import shutil
import sys
import subprocess
from pathlib import Path
import zipfile
from typing import Optional
import kagglehub
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
from helpers import PROJECT_ROOT




def duplicate_dataset_limited(src_dir, dst_dir, max_files_per_class=5):
    """
    Copie la structure de dossiers de src_dir vers dst_dir en ne gardant que max_files_per_class fichiers image par sous-dossier.
    
    Args:
        src_dir (str): chemin vers dataset source
        dst_dir (str): chemin vers dataset destination
        max_files_per_class (int): nombre max d'images à copier par sous-dossier
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(src_dir):
        # Calcul chemin relatif depuis src_dir
        rel_path = os.path.relpath(root, src_dir)
        # Nouveau chemin dans dst_dir
        target_dir = os.path.join(dst_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Filtrer uniquement fichiers images jpg/jpeg/png (en minuscules)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files = sorted(image_files)[:max_files_per_class]  # Prendre les 5 premières
        
        for file in image_files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)
            shutil.copy2(src_file, dst_file)  # copie avec métadonnées

    print(f"Copie terminée dans {dst_dir} (max {max_files_per_class} images par dossier)")



def move_dataset_if_exists(src: Path, dst: Path) -> None:
    """
    Si src existe, le déplace vers dst.
    - Crée les dossiers parents de dst si nécessaire.
    - Supprime dst existant pour éviter les conflits.
    """
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        # if dst.exists():
        #     shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"✅ Déplacé :\n  {src}\n→ {dst}")
    else:
        print(f"⚠️  Répertoire source introuvable : {src}")


def _has_kaggle_credentials() -> bool:
    """Vérifie la présence de ~/.kaggle/kaggle.json"""
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def download_with_kaggle_cli(dataset_handle: str, out_dir: Path) -> Optional[Path]:
    """
    Tente de télécharger via la CLI Kaggle si dispo et si identifiants présents.
    Retourne le chemin d'extraction si succès, sinon None.
    """
    if shutil.which("kaggle") is None:
        print("ℹ️  CLI Kaggle introuvable (pip install kaggle).")
        return None
    if not _has_kaggle_credentials():
        print("ℹ️  Identifiants Kaggle manquants (~/.kaggle/kaggle.json).")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "plantvillage.zip"
    try:
        print("⬇️  Téléchargement via Kaggle CLI…")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_handle,
                "-p",
                str(out_dir),
                "-o",
            ],
            check=True,
        )
        # Trouver le .zip téléchargé
        # La CLI nomme souvent le zip avec le slug du dataset
        candidates = list(out_dir.glob("*.zip"))
        if not candidates:
            print("⚠️  Aucun fichier ZIP trouvé après le téléchargement CLI.")
            return None
        zip_path = candidates[0]
        print(f"📦 ZIP téléchargé: {zip_path}")

        # Extraire
        extract_dir = out_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"✅ Archive extraite dans: {extract_dir}")
        return extract_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec Kaggle CLI: {e}")
        return None


def create_minimal_dataset(dst_dataset: Path, classes=("sample_class_a", "sample_class_b"), images_per_class: int = 5) -> None:
    """
    Crée un mini-dataset local avec quelques images factices pour permettre aux pipelines
    de fonctionner sans connexion Kaggle.
    """
    dst_dataset.mkdir(parents=True, exist_ok=True)
    w, h = 256, 256
    for idx, cls in enumerate(classes):
        cls_dir = dst_dataset / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new("RGB", (w, h), color=(50 + idx * 100, 150, 50 + i * 5))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"{cls} #{i+1}", fill=(255, 255, 255))
            img.save(cls_dir / f"img_{i+1:02d}.png")
    print(f"✅ Mini-dataset créé: {dst_dataset}")

if __name__ == "__main__":
    project_root = PROJECT_ROOT
    dst = project_root / "dataset" / "plantvillage" / "data"
    dst_dataset = dst / "plantvillage_5images"
    force = os.environ.get("FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes", "on")

    # Idempotence: si déjà prêt, sortir
    if dst_dataset.exists() and not force:
        print(f"⚠️  Le dataset réduit existe déjà à : {dst_dataset}")
        sys.exit(0)

    # Si on force, on nettoie la destination pour éviter les collisions de move()
    if force and dst.exists():
        print(f"♻️  FORCE_DOWNLOAD actif: suppression de {dst}")
        shutil.rmtree(dst)

    dataset_handle = "abdallahalidev/plantvillage-dataset"

    # Étape 1: tenter KaggleHub
    download_path = None
    try:
        print("⬇️  Téléchargement via KaggleHub…")
        download_path = Path(kagglehub.dataset_download(dataset_handle))
        print("Path to dataset files:", download_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        move_dataset_if_exists(download_path, dst)
    except Exception as e:
        print(f"❌ KaggleHub a échoué: {e}")
        # Étape 2: tenter Kaggle CLI si possible
        cli_extract = download_with_kaggle_cli(dataset_handle, dst)
        if cli_extract is not None:
            # Déplacer le contenu extrait dans dst si besoin
            # Si la structure extraite contient déjà le dossier "plantvillage dataset", on déplace la racine
            try:
                move_dataset_if_exists(cli_extract, dst)
            except Exception as e2:
                print(f"⚠️  Impossible de déplacer depuis la CLI: {e2}")
        else:
            print("⚠️  Utilisation d'un mini-dataset local de secours (pas d'accès Kaggle).")
            dst.mkdir(parents=True, exist_ok=True)
            create_minimal_dataset(dst_dataset, images_per_class=5)
            sys.exit(0)

    # Étape 3: déterminer la racine des images dans le dossier déplacé
    src_candidate1 = dst / "plantvillage dataset"
    if src_candidate1.exists():
        src_dataset = src_candidate1
    else:
        src_dataset = dst

    # Si des sous-dossiers existent avec des images, dupliquer un échantillon
    has_subdirs = any((src_dataset / d).is_dir() for d in os.listdir(src_dataset)) if src_dataset.exists() else False
    if src_dataset.exists() and has_subdirs:
        duplicate_dataset_limited(src_dataset, dst_dataset, max_files_per_class=5)
    else:
        # En dernier recours, créer un mini jeu factice
        create_minimal_dataset(dst_dataset, images_per_class=5)

