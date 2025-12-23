
"""
export_embeddings_keras.py
--------------------------
Exports backbone embeddings (features) for train/val/test from PlantVillage/segmented,
using the SAME split & label spaces as your Keras multi-task starter.

Outputs (in OUTDIR):
- X_train.npy, y_species_train.npy, y_health_train.npy, y_disease_train.npy
- X_val.npy,   ...
- X_test.npy,  ...
- species.json, diseases.json, classes.json  (copied for convenience)

Usage:
  python export_embeddings_keras.py --data_root /path/to/PlantVillage/segmented \
      --img_size 224 --batch 128 --outdir outputs_embeddings --backbone mobilenetv3

Backbones:

Requires: tensorflow>=2.12, numpy
"""

import os, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] GPU memory growth non appliqué: {e}")

import tf_keras as keras
from tf_keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess
from tf_keras.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess

"""
parse_args()
Rôle. Lire les arguments de la ligne de commande (chemins, tailles, etc.).
Entrées. Rien (lit sys.argv).
{{ ... }}
Sorties. Un objet args avec :
data_root: chemin du dossier segmented/;
img_size: taille entrée des images (ex. 224) ;
batch: taille de lot pour export ;
outdir: dossier de sortie ;
val_split, test_split, seed: paramètres du split ;
backbone: choix du réseau (mobilenetv3 ou efficientnetv2s).
Ce quelle fait.
Crée un parser argparse.
Déclare les options avec des valeurs par défaut.
Retourne lobjet parsé pour que main() utilise.
"""

import re

# Séparateur robuste : >= 2 underscores (gère "__" et "___")
SEP = re.compile(r'__+')

def split_species_disease(class_dirname: str):
    """
    Reçoit un nom de dossier de classe PlantVillage/segmented, p.ex.:
      - "Pepper,_bell___healthy"
      - "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    Retourne (species, disease) avec:
      - disease en minuscules
      - disease == "healthy" si c'est une feuille saine
    """
    if isinstance(class_dirname, bytes):
        class_dirname = class_dirname.decode()
    name = class_dirname.strip().strip('_')
    parts = SEP.split(name, maxsplit=1)
    species = parts[0].strip().strip('_')
    disease = (parts[1] if len(parts) > 1 else "healthy").strip().strip('_').lower()
    if disease in {"", "-", "none"}:
        disease = "healthy"
    return species, disease


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--outdir", type=str, default="outputs_embeddings")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--test_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default="mobilenetv3", choices=["mobilenetv3","efficientnetv2s"])
    # New: reuse existing splits/labels and/or extract from a finetuned Keras model
    ap.add_argument("--splits_file", type=str, default=None, help="Path to a splits.json to reuse exact train/val/test file lists")
    ap.add_argument("--labels_dir", type=str, default=None, help="Directory containing species.json and diseases.json to reuse label mappings")
    ap.add_argument("--from_keras_model", type=str, default=None, help="Path to a saved Keras functional model (.keras). Embeddings will be taken from the 'dropout' penultimate layer.")
    return ap.parse_args()

"""
scan_dataset(root: str):
Rôle. Parcourir le dossier PlantVillage/segmented pour lister toutes les classes et toutes les images.
Entrées. root = chemin vers segmented/.
Sorties.
classes: noms de dossiers (ex. "Apple___Black_rot", "Apple___healthy"…) ;
species: liste triée des espèces (ex. ["Apple","Blueberry",…]) ;
disease_names: liste triée des maladies (on exclut "healthy") ;
items: liste de tuples (chemin_image, nom_classe) pour toutes les images.
Étapes.
Liste les sous-dossiers de root = classes fines.
À partir des noms de dossiers, sépare espèce et maladie avec split("___", 1).
Récupère toutes les images (extensions .jpg/.png/.bmp) avec leur classe.
Retourne les 4 structures ci-dessus.
Pourquoi ? On a besoin :
(a) un inventaire des images,
(b) des listes de labels (espèces/maladies) pour indexer les étiquettes plus tard.
"""
def scan_dataset(root: str):
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # classes = sous-dossiers immédiats (PlantVillage/segmented/<classe>)
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    classes = [p.name for p in class_dirs]

    # Parse robuste des species/diseases à partir des noms de classes
    species_set = set()
    disease_set = set()  # on exclura 'healthy'
    for c in classes:
        sp, dis = split_species_disease(c)
        species_set.add(sp)
        if dis != "healthy":
            disease_set.add(dis)

    species = sorted(list(species_set))
    disease_names = sorted(list(disease_set))  # healthy n'est PAS inclus

    # Inventaire des images
    items = []
    for cdir in class_dirs:
        for p in cdir.rglob("*"):
            if p.suffix.lower() in exts:
                items.append((str(p), cdir.name))  # (filepath, class_name)

    return classes, species, disease_names, items


"""
stratified_split(items, val_ratio, test_ratio, seed=42)
Rôle. Découper stratifié par classe en train/val/test (pour garder les proportions).
Entrées.
items: la liste (filepath, classe) ;
val_ratio, test_ratio: parts validation/test ;
seed: graine de hasard.
Sorties. Trois listes : train_items, val_items, test_items.
Étapes.
Regroupe les images par classe fine.
Pour chaque classe : mélange, coupe un morceau pour test, un morceau pour val, le reste en train.
Concatène tout et renvoie.
Pourquoi ? Éviter que une classe rare se retrouve entièrement en train (ou entièrement en test).
"""

def stratified_split(items, val_ratio, test_ratio, seed=42):
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for fp, cls in items:
        by_class[cls].append((fp, cls))
    train, val, test = [], [], []
    for cls, lst in by_class.items():
        lst = np.array(lst, dtype=object)
        rng.shuffle(lst)
        n = len(lst)
        n_test = int(round(n*test_ratio))
        n_val  = int(round(n*val_ratio))
        test.extend(lst[:n_test].tolist())
        val.extend(lst[n_test:n_test+n_val].tolist())
        train.extend(lst[n_test+n_val:].tolist())
    return train, val, test

"""
build_backbone(name, img_size):
Rôle. Créer le modèle extraction de caractéristiques (le backbone) pré-entraîné ImageNet.
Entrées.
name: "mobilenetv3" ou "efficientnetv2s" ;
img_size: taille entrée (ex. 224).
Sorties. Un modèle Keras qui prend une image (H,W,3) et renvoie un vecteur embedding (grâce au pooling="avg").
Étapes.
Charge architecture choisie sans la tête de classification (include_top=False) et poids ImageNet.
Ajoute une entrée Keras et appelle le backbone en mode gelé (training=False) pour la stabilité.
Retourne le modèle “extracteur de features”.
Pourquoi pooling="avg" ? Pour aplatir la carte de features en un vecteur fixe (plus simple pour SVM).
"""

def build_backbone(name, img_size):
    if name=="mobilenetv3":
        base = keras.applications.MobileNetV3Large(
            include_top=False, weights="imagenet",
            input_shape=(img_size,img_size,3), pooling="avg"
        )
    else:
        base = keras.applications.EfficientNetV2S(
            include_top=False, weights="imagenet",
            input_shape=(img_size,img_size,3), pooling="avg"
        )
    inp = keras.Input(shape=(img_size,img_size,3))
    x = base(inp, training=False)
    model = keras.Model(inp, x, name=f"{name}_feature_extractor")
    return model

def build_extractor_from_saved_keras(model_path: str):
    """
    Load a saved Functional Keras model (from mt_trainer_clean.py) and return a model that outputs
    the penultimate embedding (the 'dropout' layer output). Fallbacks to the backbone pooled output
    if 'dropout' is not found.
    """
    model = keras.models.load_model(model_path)
    try:
        feat = model.get_layer("dropout").output
    except Exception:
        # Fallback: try known backbone layer names
        try:
            feat = model.get_layer("efficientnetv2-s").output
        except Exception:
            try:
                feat = model.get_layer("MobilenetV3large").output
            except Exception as e:
                raise ValueError(f"Could not locate a feature layer in saved model: {e}")
    extractor = keras.Model(inputs=model.input, outputs=feat, name="feature_extractor_from_keras")
    return extractor

"""
preprocess_image(path, img_size, preprocess_fn):
Rôle. Charger une image depuis son chemin et la préparer pour le réseau.
Entrées. path: chemin du fichier ; img_size: côté de redimensionnement.
Sorties. Un tenseur image float32 prétraité ImageNet (taille (img_size, img_size, 3)).
Étapes.
Lit le fichier (TensorFlow I/O).
Décodage RGB.
Redimensionnement bilinéaire.
Prétraitement ImageNet adapté au backbone (keras.applications.*.preprocess_input).
Pourquoi ? Chaque backbone Keras attend un prétraitement spécifique (ex. MobileNetV3: [-1,1]; EfficientNetV2: normalisation type Torch avec moyenne/écart-type).
"""

def preprocess_image(path, img_size, preprocess_fn):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size,img_size))
    img = tf.cast(img, tf.float32)
    # IMPORTANT: Utiliser le prétraitement ImageNet adapté au backbone
    # (ex: MobileNetV3 -> [-1,1], EfficientNetV2 -> normalisation spécifique)
    img = preprocess_fn(img)
    return img

"""
make_ds(items, img_size, batch, preprocess_fn):
Rôle. Construire un tf.data.Dataset efficace pour par lots : images → (espèce, maladie).
Entrées. items (liste (filepath, classe)), img_size, batch, preprocess_fn (keras.applications.*.preprocess_input).
Sorties. Un Dataset qui émet batches de (image, species, disease).
Étapes.
Crée un dataset à partir des chemins et noms de classe.
Dans _map :
appelle preprocess_image avec preprocess_fn (pour image) ;
sépare classe en species et disease via tf.strings.split(cls, _).
Batch + prefetch pour accélérer.
Note. Ici on ne fait pas de augmentation (on veut des features stables pour SVM).
"""

def make_ds(items, img_size, batch, preprocess_fn):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((
        [fp for fp, _ in items],
        [cls for _, cls in items],
    ))

    def _parse_cls_numpy(cls_str: bytes):
        sp, dis = split_species_disease(cls_str)
        return np.array(sp), np.array(dis)  # dtype=object -> tf.string

    def _map(fp, cls):
        img = preprocess_image(fp, img_size, preprocess_fn)
        sp, dis = tf.numpy_function(_parse_cls_numpy, [cls], Tout=[tf.string, tf.string])
        sp.set_shape([]); dis.set_shape([])
        return img, sp, dis

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)
    return ds


"""
main():
Rôle. Point d’entrée — enchaîne tout et sauve les résultats .npy.
Étapes (pas à pas).
Lit les arguments via parse_args().
Scan du dataset : récupère classes, species, diseases, items.
Sauvegarde classes.json, species.json, diseases.json (utile pour relire les indices plus tard).
Construit deux dictionnaires :
sp2idx = {species -> id} et dis2idx = {disease -> id} (on exclut "healthy" de dis2idx).
Split stratifié en train/val/test.
Construit le backbone Keras (gelé) avec build_backbone(...).
Boucle sur chaque split (“train”, “val”, “test”) :
crée le dataset make_ds(...) ;
passe chaque batch dans le backbone → récupère les embeddings feats (Numpy) ;
convertit les labels texte en indices :
y_sp = id d’espèce ;
y_hl = 0/1 (healthy → 0, diseased → 1) ;
y_dis = id de maladie si malade, sinon -1 (marqueur “pas de maladie”).
concatène tout (tous les batches) et sauve :
X_<split>.npy (features) ;
y_species_<split>.npy, y_health_<split>.npy, y_disease_<split>.npy.
Affiche des résumés de formes (ex. (NbImages, DimFeatures)), puis “Done”.
Pourquoi mettre -1 pour les sains dans y_disease ?
Parce qu’un classifieur “maladie” global n’a de sens que pour les feuilles malades. 
Plus tard, le script SVM filtrera y_disease != -1 pour entraîner/évaluer la tête “maladie”.
"""

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    classes, species_scanned, diseases_scanned, items = scan_dataset(args.data_root)
    print(f"[scan] #classes={len(classes)} | #species={len(species_scanned)} | #diseases(without healthy)={len(diseases_scanned)}")
    print("Exemples de classes:", classes[:3])
    # Reuse label mappings if provided
    if args.labels_dir:
        try:
            with open(os.path.join(args.labels_dir, "species.json"), "r") as f:
                species = json.load(f)
            with open(os.path.join(args.labels_dir, "diseases.json"), "r") as f:
                diseases = json.load(f)
            print(f"[labels] Reusing label mappings from {args.labels_dir} → #species={len(species)} | #diseases={len(diseases)}")
        except Exception as e:
            print(f"[WARN] Could not load labels from {args.labels_dir}: {e}. Falling back to scanned labels.")
            species = species_scanned
            diseases = diseases_scanned
    else:
        species = species_scanned
        diseases = diseases_scanned
    # Persist the labels actually used
    with open(os.path.join(args.outdir,"classes.json"),"w") as f: json.dump(classes,f,indent=2)
    with open(os.path.join(args.outdir,"species.json"),"w") as f: json.dump(species,f,indent=2)
    with open(os.path.join(args.outdir,"diseases.json"),"w") as f: json.dump(diseases,f,indent=2)
    sp2idx = {s:i for i,s in enumerate(species)}
    dis2idx = {d:i for i,d in enumerate(diseases)}  # excludes "healthy"

    # Reuse explicit splits if provided
    if args.splits_file and os.path.exists(args.splits_file):
        with open(args.splits_file, "r") as f:
            splits = json.load(f)
        
        # Helper to convert splits to (filepath, class_name) format
        def parse_split_items(split_list):
            result = []
            for x in split_list:
                if isinstance(x, str):
                    # Simple filepath: extract class_name from parent directory
                    filepath = x
                    class_name = Path(filepath).parent.name
                    result.append((filepath, class_name))
                else:
                    # Already a tuple/list
                    result.append(tuple(x))
            return result
        
        train_items = parse_split_items(splits.get("train", []))
        val_items   = parse_split_items(splits.get("val", []))
        test_items  = parse_split_items(splits.get("test", []))
        print(f"[splits] Loaded from {args.splits_file} → train {len(train_items)} | val {len(val_items)} | test {len(test_items)}")
        # Save a copy for traceability
        with open(os.path.join(args.outdir, "splits.json"), "w") as f:
            json.dump(splits, f, indent=2)
    else:
        train_items, val_items, test_items = stratified_split(items, args.val_split, args.test_split, seed=args.seed)
        print(f"Split → train {len(train_items)} | val {len(val_items)} | test {len(test_items)}")
        with open(os.path.join(args.outdir, "splits.json"), "w") as f:
            json.dump({"train": train_items, "val": val_items, "test": test_items}, f, indent=2)

    # Build feature extractor: either from a saved Keras model (finetuned) or from ImageNet-pretrained backbone
    if args.from_keras_model:
        print(f"[extractor] Using saved Keras model: {args.from_keras_model}")
        model = build_extractor_from_saved_keras(args.from_keras_model)
    else:
        model = build_backbone(args.backbone, args.img_size)
    # Choisir la fonction de prétraitement ImageNet adaptée au backbone
    if args.backbone == "mobilenetv3":
        preprocess_fn = mobilenet_v3_preprocess
    else:
        # EfficientNetV2S dans tf_keras inclut déjà un prétraitement interne (couche 'rescaling').
        # Pour éviter un double-prétraitement, on ne fait rien côté pipeline.
        preprocess_fn = (lambda img: img)
    for split_name, split_items in [("train",train_items),("val",val_items),("test",test_items)]:
        ds = make_ds(split_items, args.img_size, args.batch, preprocess_fn)
        feats_list, sp_list, hl_list, dis_list = [], [], [], []
        for imgs, sp, dis in ds:
            feats = model(imgs, training=False).numpy()
            feats_list.append(feats)
            sp_idx = np.array([sp2idx[s.decode()] for s in sp.numpy().tolist()], dtype=np.int64)
            is_diseased = np.array([1 if (d.decode()!="healthy") else 0 for d in dis.numpy().tolist()], dtype=np.int64)
            # disease index only for diseased; else -1
            dis_idx = np.array([dis2idx[d.decode()] if d.decode()!="healthy" else -1 for d in dis.numpy().tolist()], dtype=np.int64)
            sp_list.append(sp_idx)
            hl_list.append(is_diseased)
            dis_list.append(dis_idx)
        X = np.concatenate(feats_list, axis=0)
        y_sp = np.concatenate(sp_list, axis=0)
        y_hl = np.concatenate(hl_list, axis=0)
        y_dis = np.concatenate(dis_list, axis=0)
        np.save(os.path.join(args.outdir, f"X_{split_name}.npy"), X)
        np.save(os.path.join(args.outdir, f"y_species_{split_name}.npy"), y_sp)
        np.save(os.path.join(args.outdir, f"y_health_{split_name}.npy"), y_hl)
        np.save(os.path.join(args.outdir, f"y_disease_{split_name}.npy"), y_dis)
        print(f"[{split_name}] features: {X.shape}, species labels: {y_sp.shape}, health: {y_hl.shape}, disease: {y_dis.shape}")
    print("Done. Embeddings exported.")

if __name__ == "__main__":
    main()
"""
X_train.npy : matrice (N_train, D) avec D = dimension des embeddings du backbone.
y_species_train.npy : vecteur d’entiers (N_train,) → l’indice d’espèce.
y_health_train.npy : vecteur binaire (N_train,) → 0=healthy, 1=diseased.
y_disease_train.npy : vecteur d’entiers (N_train,) → indice de maladie si malade, -1 sinon.
Même chose pour val et test.
"""