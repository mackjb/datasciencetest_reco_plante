import os
import shutil
import kagglehub

def move_dataset_if_exists(src_dir: str, dst_dir: str) -> None:
    """
    Si src_dir existe, le déplace vers dst_dir.
    - Crée les dossiers parents de dst_dir si nécessaire.
    - Supprime dst_dir existant pour éviter les conflits.
    """
    if os.path.exists(src_dir):
        # Crée l'arborescence parent de dst_dir
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        # Supprime dst_dir existant (optionnel)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        # Effectue le déplacement
        shutil.move(src_dir, dst_dir)
        print(f"✅ Déplacé :\n  {src_dir}\n→ {dst_dir}")
    else:
        print(f"⚠️  Répertoire source introuvable : {src_dir}")

if __name__ == "__main__":
    # 1. Télécharge la dernière version
    path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print("Path to dataset files:", path)

    # 2. Définissez vos chemins
    src = "/home/codespace/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3"
    dst = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/plantvillage-dataset"

    # 3. Déplace si le dossier existe
    move_dataset_if_exists(src, dst)