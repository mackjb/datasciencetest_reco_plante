import shutil
from pathlib import Path
import kagglehub

def move_dataset_if_exists(src: Path, dst: Path) -> None:
    """
    Si src existe, le déplace vers dst.
    - Crée les dossiers parents de dst si nécessaire.
    - Supprime dst existant pour éviter les conflits.
    """
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"✅ Déplacé :\n  {src}\n→ {dst}")
    else:
        print(f"⚠️  Répertoire source introuvable : {src}")

if __name__ == "__main__":
    # 1. Télécharge la dernière version du dataset
    download_path = Path(kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"))
    print("Path to dataset files:", download_path)

    # 2. Définissez la racine du projet (dossier courant du script)
    project_root = Path(__file__).resolve().parent.parent

    # 3. Chemin relatif vers le cache KaggleHub et la version à déplacer
    src = project_root / ".cache" / "kagglehub" / "datasets" / "abdallahalidev" / "plantvillage-dataset" / "versions" / "3"

    # 4. Chemin relatif de destination dans votre projet
    dst = project_root / "dataset" / "plantvillage" / "plantvillage-dataset"

    # 5. Déplace si le dossier existe
    move_dataset_if_exists(src, dst)
