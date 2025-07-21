import sys
import hashlib
import requests
import tarfile
from pathlib import Path

# Ajout du chemin vers src pour importer PROJECT_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
from helpers import PROJECT_ROOT

def download_file(url: str, local_path: Path) -> None:
    """
    Télécharge un fichier en streaming.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with local_path.open('wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                print(f"\rTéléchargé : {downloaded/1024/1024:.1f} Mo / {total/1024/1024:.1f} Mo", end='')
    print("\nTéléchargement terminé.")

def check_md5(path: Path, expected_md5: str) -> bool:
    """
    Calcule et compare le MD5 d’un fichier.
    """
    hash_md5 = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    actual_md5 = hash_md5.hexdigest()
    if actual_md5.lower() == expected_md5.lower():
        print("✔ MD5 vérifié.")
        return True
    else:
        print(f"✖ MD5 mismatch : attendu {expected_md5}, obtenu {actual_md5}")
        return False

def extract_and_cleanup(archive_path: Path, extract_dir: Path) -> None:
    """
    Décompresse et supprime l’archive.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=extract_dir)
    print(f"Archive extraite dans « {extract_dir} ».")
    archive_path.unlink()
    print(f"Fichier {archive_path.name} supprimé.")

if __name__ == "__main__":
    project_root = PROJECT_ROOT
    base_dir = project_root / "dataset" / "flavia"
    archive_path = base_dir / "Leaves.tar.bz2"
    extract_dir = base_dir / "data"
    expected_md5 = "8d3ca661e201f4eac8d0975e7b6b5853"

    if extract_dir.exists():
        print(f"⚠️  Le dataset existe déjà à : {extract_dir}")
    else:
        # 1. Télécharger l’archive
        download_file(
            "https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download",
            archive_path,
        )
        # 2. Vérifier le MD5
        if check_md5(archive_path, expected_md5):
            # 3. Extraire puis supprimer l’archive
            extract_and_cleanup(archive_path, extract_dir)
        else:
            print("Abandon de l’extraction suite à la vérification MD5 échouée.")