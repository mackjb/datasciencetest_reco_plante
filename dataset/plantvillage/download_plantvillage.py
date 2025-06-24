import os
import shutil
import sys
from pathlib import Path
import kagglehub

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

if __name__ == "__main__":
    project_root = PROJECT_ROOT
    dst = project_root / "dataset" / "plantvillage" / "data"
    if dst.exists():
        print(f"⚠️  Le dataset existe déjà à : {dst}")
    else:
        # Télécharge la dernière version du dataset
        download_path = Path(kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"))
        print("Path to dataset files:", download_path)
        # Déplace le dataset
        move_dataset_if_exists(download_path, dst)
        
        # Exemple d'utilisation
        src_dataset = dst
        dst_dataset = dst / "plantvillage_5images"
        duplicate_dataset_limited(src_dataset, dst_dataset, max_files_per_class=5)

