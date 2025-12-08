#!/usr/bin/env python3
"""
Grad-CAM Quickstart (Archi 3: mono-tâche 1 tête / 35 classes Species_State)
- Charge le modèle entraîné par mono_tache_1_tete.py
- Génère une carte Grad-CAM sur une image (fournie ou prise dans dataset/plantvillage/data/plantvillage dataset/color)
- Peut appliquer Grad-CAM en lot sur un dossier d'images

Usage (exemples):
  python comparisons/interpretability/gradcam_quickstart_archi3.py \
    --image /chemin/vers/image.jpg

  python comparisons/interpretability/gradcam_quickstart_archi3.py \
    --batch-gradcam --images-dir /chemin/vers/dossier_images
"""

import argparse
import csv
import json
import importlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from PIL import Image
import matplotlib.pyplot as plt


def get_api_for(model):
    """Retourne le module Keras à utiliser (tf_keras vs tf.keras) correspondant au modèle."""
    try:
        TFK = importlib.import_module('tf_keras')
        if isinstance(model, TFK.Model):
            return TFK
    except Exception:
        pass
    return K


# Enregistre un alias pour permettre de désérialiser d'éventuels modèles sauvegardés avec tf_keras
try:
    from tensorflow.keras.utils import register_keras_serializable as _reg

    @_reg(package='tf_keras.src.engine.functional')
    class Functional(tf.keras.Model):
        pass
except Exception:
    pass


# Dossiers par défaut
BASE = Path('/workspaces/app')
MODEL_DIR = BASE / 'outputs_mono_1head_35classes'
MODEL_PATH = MODEL_DIR / 'best_model.keras'
CLASS_INDEX_JSON = MODEL_DIR / 'class_index.json'
# Répertoire de base pour les sorties Grad-CAM
OUT_BASE = BASE / 'comparisons' / 'interpretability' / 'verif_bias_gradcam'
OUT_DIR = OUT_BASE / 'archi3_quickstart'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_device(device: str = 'auto', use_mixed: bool = False):
    gpus = tf.config.list_physical_devices('GPU')
    if device == 'cpu':
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        print('[device] CPU mode')
    elif device == 'gpu':
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"[device] GPU mode | GPUs detected: {len(gpus)}")
        else:
            print('[device] GPU requested but none detected; continuing on CPU')
    else:
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"[device] AUTO mode → using GPU | GPUs detected: {len(gpus)}")
        else:
            print('[device] AUTO mode → CPU (no GPU detected)')
    if use_mixed:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print('[device] mixed_precision=enabled (mixed_float16)')
        except Exception:
            print('[device] mixed_precision request ignored (not supported)')


def find_default_image() -> Path | None:
    """Trouve une image par défaut (priorité au dossier 'color')."""
    candidates: list[Path] = []
    roots = [
        BASE / 'dataset' / 'plantvillage' / 'data' / 'plantvillage dataset' / 'color',
        BASE / 'dataset' / 'plantvillage' / 'data' / 'plantvillage_dataset' / 'color',
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
    """Charge une image, redimensionne, retourne (tensor_batched, img_rgb_uint8)."""
    pil = Image.open(path).convert('RGB').resize(target_size, Image.BILINEAR)
    # Archi 3 : EfficientNetV2S (tf_keras) inclut un Rescaling(1/255) interne,
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
    return model.inputs[0]


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
        if ('conv' in lname or 'depthwiseconv2d' in lname or 'separableconv2d' in lname or 'conv2d' in lname):
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
    # Fallback : dernière conv connectée
    collected = list(iter_conv_layers(model))
    if not collected:
        raise ValueError("Aucune couche convolutionnelle trouvée pour Grad-CAM.")
    return collected[0]


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
    conv_out = tf.cast(conv_out, tf.float32)
    grads = tf.cast(grads, tf.float32)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_out, pooled_grads), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom
    return heatmap.numpy()


def overlay_heatmap(img_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose la heatmap (jet) sur l'image RGB. Retourne un array RGB uint8."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.get_cmap('jet')
    colored = cmap(heatmap_uint8)[:, :, :3]
    colored = (colored * 255).astype('uint8')
    if colored.shape[:2] != img_rgb_uint8.shape[:2]:
        colored = np.array(Image.fromarray(colored).resize((img_rgb_uint8.shape[1], img_rgb_uint8.shape[0]), Image.BILINEAR))
    overlay = (alpha * colored + (1 - alpha) * img_rgb_uint8).astype('uint8')
    return overlay


def load_label_map(path: Path) -> list[str] | None:
    """Charge la liste de classes depuis class_index.json (dict index→nom)."""
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
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


def load_model_compat(path: Path):
    """Charge le modèle Archi 3 en privilégiant tf_keras, avec fallback tf.keras."""
    try:
        import tf_keras as TFK
        print('[load_model_compat] trying tf_keras.models.load_model(safe_mode=False)')
        return TFK.models.load_model(path, compile=False, safe_mode=False)
    except ModuleNotFoundError:
        print('[load_model_compat] tf_keras not installed or not importable; falling back to tf.keras')
    except TypeError:
        try:
            import tf_keras as TFK
            print('[load_model_compat] trying tf_keras.models.load_model without safe_mode')
            return TFK.models.load_model(path, compile=False)
        except Exception as e:
            print(f'[load_model_compat] tf_keras models.load_model failed: {e}')
    except Exception as e:
        print(f'[load_model_compat] tf_keras models.load_model failed: {e}')

    custom_objects = {
        'Functional': tf.keras.Model,
    }
    try:
        print('[load_model_compat] trying tf.keras load (compile=False, custom_objects)')
        return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        print(f'[load_model_compat] tf.keras load failed: {e}')
        raise


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Quickstart for Archi 3 (mono-tâche 35 classes)')
    parser.add_argument('--image', type=str, default=None, help="Chemin vers une image (jpg/png)")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], help="Sélection du device d'exécution")
    parser.add_argument('--mixed-precision', action='store_true', help='Activer mixed_float16 (GPU requis pour vitesse)')
    parser.add_argument('--out-subdir', type=str, default=None, help="Nom du sous-dossier sous verif_bias_gradcam pour stocker les résultats")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--class-idx', dest='class_idx', type=int, default=None, help="Indice de classe à cibler (prioritaire sur l'argmax)")
    group.add_argument('--class-name', dest='class_name', type=str, default=None, help='Nom de classe exact à cibler (selon class_index.json)')

    parser.add_argument('--conv-layer', type=str, default=None, help='Nom exact de la couche conv à utiliser pour Grad-CAM')
    parser.add_argument('--list-conv-layers', action='store_true', help='Lister les couches conv disponibles et quitter')

    parser.add_argument('--batch-gradcam', action='store_true', help='Appliquer Grad-CAM à toutes les images de --images-dir')
    parser.add_argument('--images-dir', type=str, default=None, help='Dossier contenant les images pour batch Grad-CAM')

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Modèle introuvable: {MODEL_PATH}")

    configure_device(args.device, args.mixed_precision)

    model = load_model_compat(MODEL_PATH)

    img_input = get_image_input(model)
    h = int(img_input.shape[1]) if img_input.shape[1] is not None else 224
    w = int(img_input.shape[2]) if img_input.shape[2] is not None else 224

    classes = load_label_map(CLASS_INDEX_JSON)
    if not classes:
        raise SystemExit(f"Impossible de charger les classes depuis: {CLASS_INDEX_JSON}")
    num_classes = int(img_input.shape[-1]) if img_input.shape[-1] is not None else len(classes)
    label_map = classes

    global OUT_DIR
    if args.out_subdir:
        OUT_DIR = OUT_BASE / args.out_subdir
    else:
        OUT_DIR = OUT_BASE / 'archi3_quickstart'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'list_conv_layers', False):
        print("Couches convolutionnelles disponibles (de la plus profonde vers l'entrée):")
        for l in iter_conv_layers(model):
            try:
                print(f"- {l.name} | shape: {l.output.shape}")
            except Exception:
                print(f"- {getattr(l, 'name', str(l))}")
        return

    api = get_api_for(model)
    target_out = model.outputs[0]
    infer_model = api.Model(inputs=model.inputs, outputs=target_out)

    # Mode batch Grad-CAM
    if args.batch_gradcam and args.images_dir:
        root = Path(args.images_dir)
        if not root.exists():
            raise SystemExit(f"Dossier introuvable: {root}")
        exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        images = []
        for ext in exts:
            images += list(root.rglob(ext))
        if not images:
            raise SystemExit(f"Aucune image trouvée sous: {root}")

        print(f"[batch-gradcam] Nb images: {len(images)} | classes={len(label_map)}")

        pred_file = OUT_DIR / 'les_predictions.csv'
        new_file = not pred_file.exists()
        with pred_file.open('a', newline='', encoding='utf-8') as f_pred:
            writer = csv.writer(f_pred)
            if new_file:
                writer.writerow([
                    'image_path', 'mode', 'head',
                    'class_idx', 'class_label', 'prob',
                    'top5_idx', 'top5_prob', 'top5_label',
                ])

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

            for img_path in images:
                try:
                    img_batched, _ = load_image(img_path, (w, h))
                    preds = infer_model(img_batched, training=False).numpy()[0]

                    if args.class_name is not None and label_map:
                        if args.class_name in label_map:
                            cls_idx = int(label_map.index(args.class_name))
                        else:
                            print(f"[batch-gradcam][warn] Classe '{args.class_name}' introuvable; utilisation de l'argmax.")
                            cls_idx = int(np.argmax(preds))
                    elif args.class_idx is not None:
                        cls_idx = int(args.class_idx)
                    else:
                        cls_idx = int(np.argmax(preds))

                    conv_local = conv
                    if conv_local is None:
                        conv_local = pick_conv_with_signal(model, target_out, img_batched, cls_idx)
                    heatmap = compute_gradcam(model, tf.constant(img_batched), conv_local, target_out, cls_idx)

                    img_full = np.array(Image.open(img_path).convert('RGB'))
                    overlay = overlay_heatmap(img_full, heatmap, alpha=0.45)

                    base_name = f"arch3_cam_{img_path.stem}"
                    Image.fromarray(img_full).save(OUT_DIR / f"{base_name}_original.png")
                    Image.fromarray(overlay).save(OUT_DIR / f"{base_name}_overlay.png")

                    top5_idx = np.argsort(preds)[-5:][::-1]
                    top5_prob = preds[top5_idx]
                    top5_label = [
                        label_map[i] if label_map and i < len(label_map) else str(i)
                        for i in top5_idx
                    ]
                    writer.writerow([
                        str(img_path),
                        'batch',
                        'combined',
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

    # Mode simple image
    img_path = Path(args.image) if args.image else find_default_image()
    if img_path is None or not img_path.exists():
        raise SystemExit(
            "Aucune image trouvée. Fournissez --image ou placez des images sous "
            "/workspaces/app/dataset/plantvillage/data/plantvillage dataset/color."
        )

    img_batched, _ = load_image(img_path, (w, h))
    preds = infer_model(img_batched, training=False).numpy()[0]

    if args.class_name is not None and label_map:
        if args.class_name in label_map:
            class_idx = int(label_map.index(args.class_name))
        else:
            raise SystemExit(f"Classe '{args.class_name}' introuvable dans le label map. Exemple: {label_map[:5]} ...")
    elif args.class_idx is not None:
        class_idx = int(args.class_idx)
    else:
        class_idx = int(np.argmax(preds))

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
    if conv is None:
        conv = pick_conv_with_signal(model, target_out, img_batched, class_idx)

    heatmap = compute_gradcam(model, tf.constant(img_batched), conv, target_out, class_idx)

    img_full = np.array(Image.open(img_path).convert('RGB'))
    overlay = overlay_heatmap(img_full, heatmap, alpha=0.45)

    base_name = f"arch3_cam_{img_path.stem}"
    Image.fromarray(img_full).save(OUT_DIR / f"{base_name}_original.png")
    Image.fromarray(overlay).save(OUT_DIR / f"{base_name}_overlay.png")

    try:
        pred_file = OUT_DIR / 'les_predictions.csv'
        new_file = not pred_file.exists()
        with pred_file.open('a', newline='', encoding='utf-8') as f_pred:
            writer = csv.writer(f_pred)
            if new_file:
                writer.writerow([
                    'image_path', 'mode', 'head',
                    'class_idx', 'class_label', 'prob',
                    'top5_idx', 'top5_prob', 'top5_label',
                ])
            top5_idx = np.argsort(preds)[-5:][::-1]
            top5_prob = preds[top5_idx]
            top5_label = [
                label_map[i] if label_map and i < len(label_map) else str(i)
                for i in top5_idx
            ]
            writer.writerow([
                str(img_path),
                'single',
                'combined',
                class_idx,
                label_map[class_idx] if label_map and class_idx < len(label_map) else str(class_idx),
                float(preds[class_idx]),
                json.dumps([int(i) for i in top5_idx.tolist()]),
                json.dumps([float(x) for x in top5_prob.tolist()]),
                json.dumps(top5_label),
            ])
    except Exception:
        pass

    top5 = np.argsort(preds)[-5:][::-1]
    print('\nGrad-CAM Quickstart — Archi 3 (mono-tâche 35 classes)')
    print(f'Modèle: {MODEL_PATH}')
    print(f'Image:  {img_path}')
    print(f'Classe cible (argmax): {class_idx}')
    if label_map and class_idx < len(label_map):
        print(f'Classe prédite: {label_map[class_idx]}')
    print('Top-5 prédictions:')
    for k in top5:
        name = label_map[k] if label_map and k < len(label_map) else str(k)
        print(f'  #{k}: {name} — {preds[k]:.4f}')
    print(f"\nFichiers créés dans: {OUT_DIR}")


if __name__ == '__main__':
    main()
