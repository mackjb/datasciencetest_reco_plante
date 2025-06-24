import shutil
import sys
from pathlib import Path
import kagglehub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
from helpers import PROJECT_ROOT

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
