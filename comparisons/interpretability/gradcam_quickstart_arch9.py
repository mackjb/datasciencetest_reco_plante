def get_api_for(model):
    """Return the Keras API module to use (tf_keras vs tf.keras) matching the model instance."""
    try:
        TFK = importlib.import_module('tf_keras')
        if isinstance(model, TFK.Model):
            return TFK
    except Exception:
        pass
    return K
#!/usr/bin/env python3
"""
Grad-CAM Quickstart (Archi 9: species→health→disease)
- Charge le modèle archi9
- Sélectionne automatiquement la tête 'disease' (si identifiable)
- Génère une carte Grad-CAM sur une image (fournie ou prise dans dataset/plantvillage/data/plantvillage dataset/color)
- Sauvegarde original, heatmap et overlay dans comparisons/interpretability/quickstart/

Usage:
  python comparisons/interpretability/gradcam_quickstart_arch9.py \
    [--image /chemin/vers/image.jpg]
"""

import argparse
import json
import csv
import shutil
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from PIL import Image
import matplotlib.pyplot as plt
import importlib

# Enregistre un alias pour permettre de désérialiser les modèles sauvegardés avec tf_keras
try:
    from tensorflow.keras.utils import register_keras_serializable as _reg
    @_reg(package='tf_keras.src.engine.functional')
    class Functional(tf.keras.Model):
        pass
except Exception:
    pass

# Dossiers par défaut
BASE = Path('/workspaces/app')
MODEL_DIR = BASE / 'outputs_arch9_species_health_to_disease'
MODEL_PATH = MODEL_DIR / 'best_model.keras'
DISEASES_JSON = MODEL_DIR / 'diseases.json'
SPECIES_JSON = MODEL_DIR / 'species.json'
# Répertoire de base pour les sorties Grad-CAM ; le sous-dossier dépendra de la tête (species/disease)
OUT_BASE = BASE / 'comparisons' / 'interpretability' / 'verif_bias_gradcam'
OUT_DIR = OUT_BASE / 'echantillons_species_calé'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_device(device: str = 'auto', use_mixed: bool = False):
    gpus = tf.config.list_physical_devices('GPU')
    if device == 'cpu':
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        print("[device] CPU mode")
    elif device == 'gpu':
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"[device] GPU mode | GPUs detected: {len(gpus)}")
        else:
            print("[device] GPU requested but none detected; continuing on CPU")
    else:
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"[device] AUTO mode → using GPU | GPUs detected: {len(gpus)}")
        else:
            print("[device] AUTO mode → CPU (no GPU detected)")
    if use_mixed:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("[device] mixed_precision=enabled (mixed_float16)")
        except Exception:
            print("[device] mixed_precision request ignored (not supported)")


def find_default_image() -> Path | None:
    """Trouve une image par défaut (priorité au dossier 'color')."""
    candidates: list[Path] = []
    roots = [
        # Chemin prioritaire fourni par l'utilisateur (avec espace dans le dossier)
        BASE / 'dataset' / 'plantvillage' / 'data' / 'plantvillage dataset' / 'color',
        # Variante sans espace si présente
        BASE / 'dataset' / 'plantvillage' / 'data' / 'plantvillage_dataset' / 'color',
        # Fallbacks éventuels
        BASE / 'dataset' / 'plantvillage' / 'data' / 'plantvillage dataset' / 'segmented',
        BASE / 'PlantVillage' / 'segmented',
    ]
    for root in roots:
        if root.exists():
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                candidates += list(root.rglob(ext))
        if candidates:
            break
    return candidates[0] if candidates else None


def load_image(path: Path, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Charge une image, redimensionne, normalise [0,1].
    Retourne (img_array_batched, img_rgb_uint8)
    """
    pil = Image.open(path).convert('RGB').resize(target_size, Image.BILINEAR)
    # Archi 9: EfficientNetV2S (tf_keras) inclut un Rescaling(1/255) interne,
    # donc on fournit ici des valeurs 0..255 (float32), comme dans le pipeline d'entraînement.
    arr = np.array(pil).astype('float32')
    batched = np.expand_dims(arr, axis=0)
    return batched, np.array(pil)


def get_image_input(model: tf.keras.Model):
    """Sélectionne l'entrée image (rang 4, canaux=3 si possible)."""
    for inp in model.inputs:
        shape = inp.shape
        if len(shape) == 4 and (shape[-1] == 3 or shape[-1] is None):
            return inp
    # fallback: premier input
    return model.inputs[0]


def find_last_conv2d(model: tf.keras.Model, target_output: tf.Tensor) -> tf.keras.layers.Layer:
    """Retourne la dernière couche conv (Conv2D/Depthwise/Separable) connectée à la sortie cible."""
    collected = []
    def collect_layers(l):
        try:
            TFK = importlib.import_module('tf_keras')
            model_types = (tf.keras.Model, K.Model, TFK.Model)
        except Exception:
            model_types = (tf.keras.Model, K.Model)
        if isinstance(l, model_types):
            for sub in l.layers:
                collect_layers(sub)
        else:
            collected.append(l)
    collect_layers(model)
    for layer in reversed(collected):
            lname = layer.__class__.__name__.lower()
            if not ("conv" in lname or "depthwiseconv2d" in lname or "separableconv2d" in lname or "conv2d" in lname):
                continue
            try:
                api = get_api_for(model)
                api.Model(inputs=model.inputs, outputs=[layer.output, target_output])
                return layer
            except Exception:
                continue
    raise ValueError("Aucune couche convolutionnelle connectée trouvée. Impossible de faire Grad-CAM.")


def iter_conv_layers(model: tf.keras.Model):
    collected = []
    def collect_layers(l):
        try:
            TFK = importlib.import_module('tf_keras')
            model_types = (tf.keras.Model, K.Model, TFK.Model)
        except Exception:
            model_types = (tf.keras.Model, K.Model)
        if isinstance(l, model_types):
            for sub in l.layers:
                collect_layers(sub)
        else:
            collected.append(l)
    collect_layers(model)
    for layer in reversed(collected):
        lname = layer.__class__.__name__.lower()
        if ("conv" in lname or "depthwiseconv2d" in lname or "separableconv2d" in lname or "conv2d" in lname):
            yield layer


def pick_conv_with_signal(model: tf.keras.Model, target_output: tf.Tensor, img_batched: np.ndarray, class_idx: int):
    for layer in iter_conv_layers(model):
        try:
            api = get_api_for(model)
            grad_model = api.Model(inputs=model.inputs, outputs=[layer.output, target_output])
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(tf.constant(img_batched), training=False)
                if preds.shape.rank != 2:
                    continue
                class_channel = preds[:, class_idx]
            grads = tape.gradient(class_channel, conv_out)
            if grads is None:
                continue
            val = tf.reduce_sum(tf.abs(grads)).numpy()
            if val > 1e-8:
                return layer
        except Exception:
            continue
    return find_last_conv2d(model, target_output)


def load_label_map(path: Path) -> list[str] | None:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                # {"0": "classA", ...} ou {"classes": [..]}
                if 'classes' in data and isinstance(data['classes'], list):
                    return data['classes']
                # Trier par clé numérique si dict indexé
                try:
                    items = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
                    return [v for _, v in items]
                except Exception:
                    return list(data.values())
            elif isinstance(data, list):
                return data
    except Exception:
        pass
    return None


def pick_disease_output(model: tf.keras.Model, diseases: list[str] | None) -> tuple[tf.Tensor, int]:
    """Sélectionne la tête 'disease' (par nom ou dimension). Retourne (tensor_sortie, nb_classes)."""
    outputs = model.outputs
    # Si les noms sont disponibles et contiennent 'disease'
    try:
        names = getattr(model, 'output_names', None)
    except Exception:
        names = None
    if names:
        for i, n in enumerate(names):
            if 'disease' in n.lower():
                return outputs[i], int(outputs[i].shape[-1])
    # Sinon, utiliser le nombre de classes attendu
    expected = len(diseases) if diseases else None
    if expected is not None:
        for out in outputs:
            if out.shape[-1] == expected:
                return out, int(expected)
    # Fallback: prendre la sortie avec la plus grande dimension de classes
    out = max(outputs, key=lambda t: int(t.shape[-1]))
    return out, int(out.shape[-1])


def compute_gradcam(model: tf.keras.Model,
                     img_tensor: tf.Tensor,
                     conv_layer: tf.keras.layers.Layer,
                     target_output: tf.Tensor,
                     class_index: int) -> np.ndarray:
    """Calcule Grad-CAM pour une classe cible. Retourne heatmap [H,W] normalisée [0,1]."""
    api = get_api_for(model)
    grad_model = api.Model(inputs=model.inputs, outputs=[conv_layer.output, target_output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        if preds.shape.rank == 2:
            class_channel = preds[:, class_index]
        else:
            raise ValueError(f"Forme de sortie inattendue: {preds.shape}")
    grads = tape.gradient(class_channel, conv_out)
    if grads is None:
        grads = tf.zeros_like(conv_out)
    # Pondération par moyenne globale des gradients
    conv_out = tf.cast(conv_out, tf.float32)
    grads = tf.cast(grads, tf.float32)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_out, pooled_grads), axis=-1)
    # ReLU et normalisation
    heatmap = tf.nn.relu(heatmap)
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom
    return heatmap.numpy()


def overlay_heatmap(img_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose la heatmap (jet) sur l'image RGB. Retourne un array RGB uint8."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.get_cmap('jet')
    colored = cmap(heatmap_uint8)[:, :, :3]  # drop alpha
    colored = (colored * 255).astype('uint8')
    # Redimension si besoin
    if colored.shape[:2] != img_rgb_uint8.shape[:2]:
        colored = np.array(Image.fromarray(colored).resize((img_rgb_uint8.shape[1], img_rgb_uint8.shape[0]), Image.BILINEAR))
    overlay = (alpha * colored + (1 - alpha) * img_rgb_uint8).astype('uint8')
    return overlay


def load_model_compat(path: Path):
    # 1) Charger via tf_keras (paquet tf-keras) — le plus compatible avec les modèles tf_keras
    try:
        import tf_keras as TFK
        print("[load_model_compat] trying tf_keras.models.load_model(safe_mode=False)")
        return TFK.models.load_model(path, compile=False, safe_mode=False)
    except ModuleNotFoundError:
        print("[load_model_compat] tf_keras not installed or not importable; falling back to tf.keras")
    except TypeError:
        # safe_mode non supporté
        try:
            import tf_keras as TFK
            print("[load_model_compat] trying tf_keras.models.load_model without safe_mode")
            return TFK.models.load_model(path, compile=False)
        except Exception as e:
            print(f"[load_model_compat] tf_keras models.load_model failed: {e}")
    except Exception as e:
        print(f"[load_model_compat] tf_keras models.load_model failed: {e}")

    # 2) Fallback: tf.keras chargeur avec custom_objects pour résoudre 'Functional'
    custom_objects = {
        'Functional': tf.keras.Model,
    }
    try:
        print("[load_model_compat] trying tf.keras load (compile=False, custom_objects)")
        return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        print(f"[load_model_compat] tf.keras load failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Quickstart for Archi 9')
    parser.add_argument('--image', type=str, default=None, help='Chemin vers une image (jpg/png)')
    parser.add_argument('--head', type=str, default='disease', choices=['disease', 'species'], help='Tête à expliquer')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','gpu'], help='Sélection du device d\'exécution')
    parser.add_argument('--mixed-precision', action='store_true', help='Activer mixed_float16 (GPU requis pour vitesse)')
    parser.add_argument('--out-subdir', type=str, default=None, help="Nom du sous-dossier sous verif_bias_gradcam pour stocker les résultats")
    # Cible de classe (mutuellement exclusif)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--class-idx', dest='class_idx', type=int, default=None, help="Indice de classe à cibler (prioritaire sur l'argmax)")
    group.add_argument('--class-name', dest='class_name', type=str, default=None, help='Nom de classe exact à cibler (selon le JSON de labels)')
    # Sélection/énumération des couches convolutionnelles
    parser.add_argument('--conv-layer', type=str, default=None, help='Nom exact de la couche conv à utiliser pour Grad-CAM')
    parser.add_argument('--list-conv-layers', action='store_true', help='Lister les couches conv disponibles et quitter')
    # Évaluation en lot et export CSV
    parser.add_argument('--eval-dir', type=str, default=None, help='Dossier racine contenant les images par sous-dossiers de classe')
    parser.add_argument('--dump-csv', type=str, default=None, help='Chemin du CSV de sortie pour enregistrer les prédictions par image')
    parser.add_argument('--confusion-filter', type=str, default=None, help="Filtre de confusion sous la forme true=<label_vrai>,pred=<label_predit>")
    parser.add_argument('--copy-errors-to', type=str, default=None, help='Dossier vers lequel copier les images filtrées')
    # Grad-CAM en lot sur un dossier d'images (modèle chargé une seule fois)
    parser.add_argument('--batch-gradcam', action='store_true', help='Appliquer Grad-CAM à toutes les images de --images-dir')
    parser.add_argument('--images-dir', type=str, default=None, help='Dossier contenant les images pour batch Grad-CAM')
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Modèle introuvable: {MODEL_PATH}")

    configure_device(args.device, args.mixed_precision)

    # Charger le modèle (sans compile pour éviter dépendances custom)
    model = load_model_compat(MODEL_PATH)

    # Déterminer l'entrée image et la taille cible
    img_input = get_image_input(model)
    h = int(img_input.shape[1]) if img_input.shape[1] is not None else 224
    w = int(img_input.shape[2]) if img_input.shape[2] is not None else 224

    # Sélection image
    img_path = Path(args.image) if args.image else find_default_image()
    if img_path is None or not img_path.exists():
        raise SystemExit(
            "Aucune image trouvée. Fournissez --image ou placez des images sous "
            "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color."
        )

    img_batched, _ = load_image(img_path, (w, h))

    # Cartes de labels
    diseases = load_label_map(DISEASES_JSON)
    species = load_label_map(SPECIES_JSON)

    # Sélection de la tête à expliquer
    if args.head == 'disease':
        target_out, num_classes = pick_disease_output(model, diseases)
        label_map = diseases
    else:
        # Heuristique: sortie avec la plus petite dimension (souvent species)
        out = min(model.outputs, key=lambda t: int(t.shape[-1]))
        target_out = out
        num_classes = int(out.shape[-1])
        label_map = species

    # Choix du dossier de sortie en fonction de la tête
    global OUT_DIR
    if args.out_subdir:
        OUT_DIR = OUT_BASE / args.out_subdir
    else:
        if args.head == 'species':
            OUT_DIR = OUT_BASE / 'echantillons_species_calé'
        else:
            OUT_DIR = OUT_BASE / 'echantillons_disease_calé'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Liste des couches conv et sortie immédiate si demandé
    if getattr(args, 'list_conv_layers', False):
        print("Couches convolutionnelles disponibles (de la plus profonde vers l'entrée):")
        for l in iter_conv_layers(model):
            try:
                print(f"- {l.name} | shape: {l.output.shape}")
            except Exception:
                print(f"- {getattr(l, 'name', str(l))}")
        return

    # Mode Grad-CAM en lot: parcourir un dossier et appliquer Grad-CAM à chaque image
    if args.batch_gradcam and args.images_dir:
        root = Path(args.images_dir)
        if not root.exists():
            raise SystemExit(f"Dossier introuvable: {root}")
        # Détermination de la classe cible (id unique pour toute la série)
        api = get_api_for(model)
        infer_model = api.Model(inputs=model.inputs, outputs=target_out)
        if args.class_name is not None and label_map:
            if args.class_name in label_map:
                class_idx = int(label_map.index(args.class_name))
            else:
                raise SystemExit(f"Classe '{args.class_name}' introuvable dans le label map. Exemple: {label_map[:5]} ...")
        elif args.class_idx is not None:
            class_idx = int(args.class_idx)
        else:
            class_idx = None  # on utilisera l'argmax par image si non spécifié

        # Choix de la couche conv à l'avance si possible
        conv = None
        if args.conv_layer:
            for layer in iter_conv_layers(model):
                if getattr(layer, 'name', '') == args.conv_layer:
                    try:
                        api.Model(inputs=model.inputs, outputs=[layer.output, target_out])
                        conv = layer
                    except Exception:
                        conv = None
                    break
            if conv is None:
                print(f"[warn] Couche '{args.conv_layer}' introuvable ou non connectée; sélection automatique activée.")

        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        images = []
        for ext in exts:
            images += list(root.rglob(ext))
        if not images:
            raise SystemExit(f"Aucune image trouvée sous: {root}")

        print(f"[batch-gradcam] Nb images: {len(images)} | head={args.head} | class={'argmax' if class_idx is None else class_idx}")

        # Prépare le fichier des prédictions
        pred_file = OUT_DIR / "les_predictions.csv"
        new_file = not pred_file.exists()
        with pred_file.open("a", newline="", encoding="utf-8") as f_pred:
            writer = csv.writer(f_pred)
            if new_file:
                writer.writerow([
                    "image_path", "mode", "head",
                    "class_idx", "class_label", "prob",
                    "top5_idx", "top5_prob", "top5_label",
                ])

            for img_path in images:
                try:
                    img_batched, _ = load_image(img_path, (w, h))
                    preds = infer_model(img_batched, training=False).numpy()[0]
                    # classe cible par image si argmax demandé
                    if class_idx is None:
                        cls_idx = int(np.argmax(preds))
                    else:
                        cls_idx = class_idx
                    # conv target (sélection paresseuse si non fixé)
                    conv_local = conv
                    if conv_local is None:
                        conv_local = pick_conv_with_signal(model, target_out, img_batched, cls_idx)
                    heatmap = compute_gradcam(model, tf.constant(img_batched), conv_local, target_out, cls_idx)

                    # Recharger l'image full-resolution pour l'overlay
                    img_full = np.array(Image.open(img_path).convert('RGB'))
                    overlay = overlay_heatmap(img_full, heatmap, alpha=0.45)

                    base_name = f"arch9_{args.head}_cam_{img_path.stem}"
                    Image.fromarray(img_full).save(OUT_DIR / f"{base_name}_original.png")
                    Image.fromarray(overlay).save(OUT_DIR / f"{base_name}_overlay.png")

                    # Log de la prédiction pour cette image
                    top5_idx = np.argsort(preds)[-5:][::-1]
                    top5_prob = preds[top5_idx]
                    top5_label = [
                        label_map[i] if label_map and i < len(label_map) else str(i)
                        for i in top5_idx
                    ]
                    writer.writerow([
                        str(img_path),
                        "batch",
                        args.head,
                        cls_idx,
                        label_map[cls_idx] if label_map and cls_idx < len(label_map) else str(cls_idx),
                        float(preds[cls_idx]),
                        json.dumps([int(i) for i in top5_idx.tolist()]),
                        json.dumps([float(x) for x in top5_prob.tolist()]),
                        json.dumps(top5_label),
                    ])
                except Exception as e:
                    print(f"[batch-gradcam][warn] Erreur sur {img_path}: {e}")

        print(f"[batch-gradcam] Fichiers créés dans: {OUT_DIR}")
        return

    # Mode évaluation: parcourir un dossier, prédire et dumper un CSV
    if args.eval_dir and args.dump_csv:
        root = Path(args.eval_dir)
        if not root.exists():
            raise SystemExit(f"Dossier introuvable: {root}")
        api = get_api_for(model)
        infer_model = api.Model(inputs=model.inputs, outputs=target_out)
        rows = []
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        images = []
        for ext in exts:
            images += list(root.rglob(ext))
        # Prépare une table de correspondance label → label officiel (insensible à la casse/séparateurs)
        def _canon(s: str) -> str:
            s = s.strip().lower()
            # unifier séparateurs espace/underscore/traits
            out = []
            prev_us = False
            for ch in s:
                if ch.isalnum():
                    out.append(ch)
                    prev_us = False
                else:
                    if not prev_us:
                        out.append('_')
                        prev_us = True
            can = ''.join(out).strip('_')
            while '__' in can:
                can = can.replace('__','_')
            return can
        official_map = {}
        if label_map:
            for lab in label_map:
                official_map[_canon(lab)] = lab
        def infer_true_label(p: Path) -> str:
            parent = p.parent.name
            if '___' in parent:
                left, right = parent.split('___', 1)
                raw = left if args.head == 'species' else right
            else:
                raw = parent
            # aligne sur label officiel si possible
            off = official_map.get(_canon(raw)) if official_map else None
            return off if off else raw
        for img_path in images:
            try:
                img_b, _ = load_image(img_path, (w, h))
                preds_vec = infer_model(img_b, training=False).numpy()[0]
                pred_idx = int(np.argmax(preds_vec))
                true_label = infer_true_label(img_path)
                pred_label = label_map[pred_idx] if label_map and pred_idx < len(label_map) else str(pred_idx)
                true_idx = label_map.index(true_label) if (label_map and true_label in label_map) else -1
                top5_idx = np.argsort(preds_vec)[-5:][::-1]
                top5_prob = preds_vec[top5_idx]
                top5_label = [label_map[i] if label_map and i < len(label_map) else str(i) for i in top5_idx]
                rows.append([
                    str(img_path), true_label, true_idx,
                    pred_label, pred_idx, float(preds_vec[pred_idx]),
                    json.dumps([int(i) for i in top5_idx.tolist()]),
                    json.dumps([float(x) for x in top5_prob.tolist()]),
                    json.dumps(top5_label),
                ])
            except Exception as e:
                rows.append([str(img_path), 'ERROR', -1, 'ERROR', -1, 0.0, '[]', '[]', json.dumps([str(e)])])
        out_csv = Path(args.dump_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path','true_label','true_idx','pred_label','pred_idx','prob_pred','top5_idx','top5_prob','top5_label'])
            writer.writerows(rows)
        print(f"CSV écrit: {out_csv} | nb images: {len(rows)}")

        # Filtrage optionnel des confusions et copie
        if args.confusion_filter:
            m_true = re.search(r"true=([^,]+)", args.confusion_filter)
            m_pred = re.search(r"pred=([^,]+)", args.confusion_filter)
            true_f = m_true.group(1) if m_true else None
            pred_f = m_pred.group(1) if m_pred else None
            if not true_f or not pred_f:
                print("[warn] --confusion-filter mal formé. Utilisez: true=<label>,pred=<label>")
            else:
                filtered = [r for r in rows if len(r) >= 6 and r[1] == true_f and r[3] == pred_f]
                # Écrire CSV filtré à côté
                safe_true = re.sub(r"[^A-Za-z0-9_()+.-]", "_", true_f)
                safe_pred = re.sub(r"[^A-Za-z0-9_()+.-]", "_", pred_f)
                filtered_csv = out_csv.with_name(out_csv.stem + f"_filtered_true={safe_true}_pred={safe_pred}" + out_csv.suffix)
                with filtered_csv.open('w', newline='', encoding='utf-8') as f2:
                    writer = csv.writer(f2)
                    writer.writerow(['image_path','true_label','true_idx','pred_label','pred_idx','prob_pred','top5_idx','top5_prob','top5_label'])
                    writer.writerows(filtered)
                print(f"CSV filtré écrit: {filtered_csv} | nb images: {len(filtered)}")

                # Copier les images si demandé
                if args.copy_errors_to:
                    dst_dir = Path(args.copy_errors_to)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    for r in filtered:
                        try:
                            src = Path(r[0])
                            dst = dst_dir / src.name
                            if not dst.exists():
                                shutil.copy2(src, dst)
                        except Exception:
                            pass
                    print(f"Images copiées: {len(filtered)} → {dst_dir}")
        return

    # Prédiction et détermination de la classe cible
    api = get_api_for(model)
    infer_model = api.Model(inputs=model.inputs, outputs=target_out)
    preds = infer_model(img_batched, training=False).numpy()[0]
    try:
        print(f"Preds shape: {preds.shape}")
    except Exception:
        pass
    if args.class_name is not None and label_map:
        if args.class_name in label_map:
            class_idx = int(label_map.index(args.class_name))
        else:
            raise SystemExit(f"Classe '{args.class_name}' introuvable dans le label map. Exemple: {label_map[:5]} ...")
    elif args.class_idx is not None:
        class_idx = int(args.class_idx)
    else:
        class_idx = int(np.argmax(preds))

    # Couche conv cible (sélection par nom si fourni, sinon fallback grad != 0)
    conv = None
    if args.conv_layer:
        for layer in iter_conv_layers(model):
            if getattr(layer, 'name', '') == args.conv_layer:
                try:
                    api = get_api_for(model)
                    api.Model(inputs=model.inputs, outputs=[layer.output, target_out])
                    conv = layer
                except Exception:
                    conv = None
                break
        if conv is None:
            print(f"[warn] Couche '{args.conv_layer}' introuvable ou non connectée; sélection automatique activée.")
    if conv is None:
        conv = pick_conv_with_signal(model, target_out, img_batched, class_idx)
    try:
        print(f"Selected head shape: {target_out.shape}")
        print(f"Selected conv layer: {getattr(conv, 'name', str(conv))} | output shape: {conv.output.shape}")
        print(f"Input image tensor shape: {img_batched.shape}")
    except Exception:
        pass

    # Calcul Grad-CAM
    heatmap = compute_gradcam(model, tf.constant(img_batched), conv, target_out, class_idx)

    # Recharger l'image full-resolution pour l'overlay
    img_full = np.array(Image.open(img_path).convert('RGB'))
    overlay = overlay_heatmap(img_full, heatmap, alpha=0.45)

    # Sauvegardes (full-resolution)
    base_name = f"arch9_{args.head}_cam_{img_path.stem}"
    Image.fromarray(img_full).save(OUT_DIR / f"{base_name}_original.png")
    Image.fromarray(overlay).save(OUT_DIR / f"{base_name}_overlay.png")

    # Heatmap colorée avec barre de couleurs (désactivée : on ne conserve que original et overlay)

    # Sauvegarde de la prédiction de cette image dans les_predictions.csv
    try:
        pred_file = OUT_DIR / "les_predictions.csv"
        new_file = not pred_file.exists()
        with pred_file.open("a", newline="", encoding="utf-8") as f_pred:
            writer = csv.writer(f_pred)
            if new_file:
                writer.writerow([
                    "image_path", "mode", "head",
                    "class_idx", "class_label", "prob",
                    "top5_idx", "top5_prob", "top5_label",
                ])
            top5_idx = np.argsort(preds)[-5:][::-1]
            top5_prob = preds[top5_idx]
            top5_label = [
                label_map[i] if label_map and i < len(label_map) else str(i)
                for i in top5_idx
            ]
            writer.writerow([
                str(img_path),
                "single",
                args.head,
                class_idx,
                label_map[class_idx] if label_map and class_idx < len(label_map) else str(class_idx),
                float(preds[class_idx]),
                json.dumps([int(i) for i in top5_idx.tolist()]),
                json.dumps([float(x) for x in top5_prob.tolist()]),
                json.dumps(top5_label),
            ])
    except Exception:
        pass

    # Affichage console
    top5 = np.argsort(preds)[-5:][::-1]
    print("\nGrad-CAM Quickstart — Archi 9")
    print(f"Modèle: {MODEL_PATH}")
    print(f"Image:  {img_path}")
    print(f"Tête:   {args.head} | Classe cible (argmax): {class_idx}")
    if label_map and class_idx < len(label_map):
        print(f"Classe prédite: {label_map[class_idx]}")
    print("Top-5 prédictions:")
    for k in top5:
        name = label_map[k] if label_map and k < len(label_map) else str(k)
        print(f"  #{k}: {name} — {preds[k]:.4f}")
    print(f"\nFichiers créés dans: {OUT_DIR}")


if __name__ == '__main__':
    main()
