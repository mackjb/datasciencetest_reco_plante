# ============================================================
# 🚀 Script YOLOv8 - Classification PlantVillage + Analyse complète
# ============================================================

# -------------------------------
# 0️⃣ Pré-requis
# -------------------------------
# Prérequis (à installer une fois dans votre environnement):
# pip install ultralytics tqdm scikit-learn pandas matplotlib

import os, random, shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, top_k_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import logging
import yaml
try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# -------------------------------
# 🔧 Chargement de la configuration YAML (optionnel)
# -------------------------------
CONFIG = {}
try:
    # Remonter à la racine du repo et lire configs/default.yaml
    REPO_ROOT = Path(__file__).resolve().parents[3]
    CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            CONFIG = yaml.safe_load(f) or {}
        logging.info(f"Configuration chargée depuis {CONFIG_PATH}")
except Exception as e:
    logging.warning(f"Impossible de charger la configuration YAML: {e}")

paths_cfg = CONFIG.get("paths", {})
yolo_cfg = CONFIG.get("yolov8", {})

# -------------------------------
# 1️⃣ Chemin du dataset
# -------------------------------
DATA_DIR = Path(paths_cfg.get(
    "dataset_dir",
    "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented",
))
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset introuvable : {DATA_DIR}")
print("✅ Dataset trouvé :", DATA_DIR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Dataset: {DATA_DIR}")

# -------------------------------
# 2️⃣ Nettoyage des fichiers corrompus
# -------------------------------
removed = 0
for root, dirs, files_in_dir in os.walk(DATA_DIR):
    for f in tqdm(files_in_dir, desc=f"Vérif {root}"):
        path = os.path.join(root, f)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("❌ Fichier corrompu supprimé :", path)
            os.remove(path)
            removed += 1
print(f"✅ Nettoyage terminé. Fichiers corrompus supprimés : {removed}")

# -------------------------------
# 3️⃣ Vérification des classes
# -------------------------------
classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
print("✅ Classes détectées :", classes)
logging.info(f"Classes détectées ({len(classes)}): {classes}")

# -------------------------------
# 4️⃣ Split train/valid
# -------------------------------
BASE_DIR = Path(paths_cfg.get(
    "processed_dir",
    "/workspaces/datasciencetest_reco_plante/data/PlantVillage_Processed",
))
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "valid"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

# Reproductibilité
SEED = int(yolo_cfg.get("seed", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

VAL_SPLIT = 0.2
REUSE_SPLIT_IF_EXISTS = True
for class_folder in DATA_DIR.iterdir():
    if class_folder.is_dir():
        images = [p for p in class_folder.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']]
        if not images:
            print(f"⚠️ Aucun fichier dans {class_folder}")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * VAL_SPLIT)
        val_images = images[:split_idx]
        train_images = images[split_idx:]

        (TRAIN_DIR / class_folder.name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / class_folder.name).mkdir(parents=True, exist_ok=True)

        # Si les dossiers cibles contiennent déjà des images et que l'option est activée, on saute la recopie
        if REUSE_SPLIT_IF_EXISTS:
            train_target = TRAIN_DIR / class_folder.name
            val_target = VAL_DIR / class_folder.name
            if any(train_target.glob('*')) and any(val_target.glob('*')):
                print(f"⏭️ Split déjà présent pour {class_folder.name}, on réutilise.")
                continue

        for img in train_images:
            shutil.copy(str(img), str(TRAIN_DIR / class_folder.name / img.name))
        for img in val_images:
            shutil.copy(str(img), str(VAL_DIR / class_folder.name / img.name))

print("✅ Dataset train/valid prêt.")
logging.info(f"Split effectué dans: {BASE_DIR}")

# -------------------------------
# 5️⃣ Entraînement YOLOv8
# -------------------------------
RESULTS_BASE = Path(paths_cfg.get(
    "results_dir",
    "/workspaces/datasciencetest_reco_plante/results",
))
RESULTS_BASE.mkdir(parents=True, exist_ok=True)
PROJECT_NAME = str(yolo_cfg.get("project_name", "yolov8_segmented_finetune"))
results_dir = RESULTS_BASE / PROJECT_NAME

# Configure logging to file
log_file = results_dir / "train.log"
results_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_file) for h in logger.handlers):
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
logging.info("Démarrage entraînement YOLOv8")
model = YOLO("yolov8n-cls.pt")
history = model.train(
    data=str(BASE_DIR),
    epochs=int(yolo_cfg.get("epochs", 30)),
    batch=int(yolo_cfg.get("batch", 16)),
    imgsz=int(yolo_cfg.get("imgsz", 224)),
    patience=int(yolo_cfg.get("patience", 5)),
    project=str(RESULTS_BASE),
    name=PROJECT_NAME,
    exist_ok=True
)
print("✅ Entraînement terminé :", results_dir)
logging.info(f"Entraînement terminé. Résultats: {results_dir}")

# -------------------------------
# 6️⃣ Visualisation des pertes (loss)
# -------------------------------
results_csv = results_dir / "results.csv"
if results_csv.exists():
    df_results = pd.read_csv(results_csv)
    print("\n📈 Dernières métriques d'entraînement :")
    print(df_results.tail().to_string(index=False))
    logging.info("Dernières métriques:\n" + df_results.tail().to_string(index=False))

    plt.figure(figsize=(8,5))
    plt.plot(df_results["epoch"], df_results["train/loss"], label="Train Loss")
    plt.plot(df_results["epoch"], df_results["val/loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Évolution de la perte (Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curves.png", dpi=150)
    plt.show()
else:
    print("⚠️ Fichier results.csv introuvable.")

# -------------------------------
# 6️⃣ BIS : Détection et visualisation du surapprentissage
# -------------------------------
if results_csv.exists():
    print("\n🧠 Analyse du surapprentissage...")

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(df_results["epoch"], df_results["train/loss"], label="Train Loss")
    plt.plot(df_results["epoch"], df_results["val/loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbes de perte")
    plt.legend()
    plt.grid(True)

    if "train/acc" in df_results.columns and "val/acc" in df_results.columns:
        plt.subplot(1,2,2)
        plt.plot(df_results["epoch"], df_results["train/acc"], label="Train Accuracy")
        plt.plot(df_results["epoch"], df_results["val/acc"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Courbes d'accuracy")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "loss_acc_curves.png", dpi=150)
    plt.show()

    if "train/acc" in df_results.columns and "val/acc" in df_results.columns:
        df_results["overfit_gap"] = df_results["train/acc"] - df_results["val/acc"]

        plt.figure(figsize=(8,4))
        plt.bar(df_results["epoch"], df_results["overfit_gap"])
        plt.xlabel("Epoch")
        plt.ylabel("Train - Val Accuracy")
        plt.title("Écart d'accuracy (indicateur de surapprentissage)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "overfit_gap.png", dpi=150)
        plt.show()

        last_gap = df_results["overfit_gap"].iloc[-1]
        if last_gap > 0.1:
            print(f"⚠️ Surapprentissage détecté : le modèle est {last_gap:.2%} meilleur sur le train que sur la validation.")
            logging.warning(f"Surapprentissage détecté : le modèle est {last_gap:.2%} meilleur sur le train que sur la validation.")
        else:
            print("✅ Aucun signe fort de surapprentissage détecté.")
            logging.info("Aucun signe fort de surapprentissage détecté.")
    else:
        print("⚠️ Accuracy non disponible (selon version YOLO).")
        logging.warning("Accuracy non disponible dans results.csv (colonnes train/acc, val/acc).")
        
# -------------------------------
# 7️⃣ Évaluation détaillée + sauvegarde pour LIME/SHAP
# -------------------------------
print("📊 Évaluation détaillée du modèle...")
logging.info("Évaluation détaillée sur la validation...")

best_model_path = results_dir / "weights/best.pt"
best_model = YOLO(best_model_path)

# Création liste validation
val_images = []
val_labels = []
for idx, cls in enumerate(classes):
    img_paths = list((VAL_DIR / cls).glob("*"))
    val_images.extend(img_paths)
    val_labels.extend([idx] * len(img_paths))

y_true = np.array(val_labels)
y_pred = []
y_probs = []

for img_path in tqdm(val_images, desc="🔍 Prédictions validation"):
    results = best_model(img_path, verbose=False)
    probs = results[0].probs.data.cpu().numpy()
    y_probs.append(probs)
    y_pred.append(np.argmax(probs))

y_probs = np.array(y_probs)
y_pred = np.array(y_pred)

# Rapport de classification
report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
top1 = top_k_accuracy_score(y_true, y_probs, k=1)
top5 = top_k_accuracy_score(y_true, y_probs, k=5)

df_report = pd.DataFrame(report).transpose()
df_report["top1_accuracy"] = top1
df_report["top5_accuracy"] = top5

print("\n✅ Résultats par classe :\n")
print(df_report.round(3).to_string())
logging.info("Rapport de classification:\n" + df_report.round(3).to_string())

# Sauvegarde
csv_report = results_dir / "classification_report.csv"
df_report.to_csv(csv_report, index=True)
logging.info(f"Rapport sauvegardé: {csv_report}")

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
plt.figure(figsize=(max(8, len(classes)*0.4), max(6, len(classes)*0.4)))
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion')
plt.colorbar(im, fraction=0.046, pad=0.04)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2. if cm.max() > 0 else 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('Vrai label')
plt.xlabel('Label prédit')
plt.tight_layout()
plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
plt.show()

# Sauvegarde des prédictions pour LIME/SHAP
preds_df = pd.DataFrame({
    "image": [str(p) for p in val_images],
    "true_label": [classes[i] for i in y_true],
    "pred_label": [classes[i] for i in y_pred]
})
probs_df = pd.DataFrame(y_probs, columns=classes)
probs_full = pd.concat([preds_df, probs_df], axis=1)

csv_probs = results_dir / "predictions_probs.csv"
probs_full.to_csv(csv_probs, index=False)

print(f"📁 Rapport métriques : {csv_report}")
print(f"📁 Probabilités (pour LIME/SHAP) : {csv_probs}")
logging.info(f"Fichiers sauvegardés -> rapport: {csv_report}, probs: {csv_probs}")

# -------------------------------
# 8️⃣ Téléchargement des fichiers utiles
# -------------------------------
if IN_COLAB:
    files.download(str(csv_report))
    files.download(str(csv_probs))
    files.download(str(best_model_path))
    print("✅ Fichiers exportés : rapport, probabilités, poids du modèle (Colab).")
else:
    print("✅ Fichiers prêts localement :")
    print(f" - Rapport métriques : {csv_report}")
    print(f" - Probabilités : {csv_probs}")
    print(f" - Meilleurs poids : {best_model_path}")
