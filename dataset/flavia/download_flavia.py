import requests
import hashlib
import tarfile
import os

def download_file(url, local_path):
    """Télécharge un fichier en streaming."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                print(f"\rTéléchargé : {downloaded/1024/1024:.1f} Mo / {total/1024/1024:.1f} Mo", end='')
    print("\nTéléchargement terminé.")

def check_md5(path, expected_md5):
    """Calcule et compare le MD5 d’un fichier."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    actual_md5 = hash_md5.hexdigest()
    if actual_md5.lower() == expected_md5.lower():
        print("✔ MD5 vérifié.")
        return True
    else:
        print(f"✖ MD5 mismatch : attendu {expected_md5}, obtenu {actual_md5}")
        return False

def extract_and_cleanup(archive_path, extract_dir):
    """Décompresse et supprime l’archive."""
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=extract_dir)
    print(f"Archive extraite dans « {extract_dir} ».")
    os.remove(archive_path)
    print(f"Fichier {os.path.basename(archive_path)} supprimé.")

if __name__ == "__main__":
    # URL de téléchargement (SourceForge mirror)
    url = "https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download"
    archive_path = "/workspaces/datasciencetest_reco_plante/dataset/flavia/Leaves.tar.bz2"
    extract_dir = "/workspaces/datasciencetest_reco_plante/dataset/flavia/flavia-dataset"
    expected_md5 = "8d3ca661e201f4eac8d0975e7b6b5853"

    # 1. Télécharger l’archive
    download_file(url, archive_path)

    # 2. Vérifier le MD5
    if check_md5(archive_path, expected_md5):
        # 3. Extraire puis supprimer l’archive
        extract_and_cleanup(archive_path, extract_dir)
    else:
        print("Abandon de l’extraction suite à la vérification MD5 échouée.")