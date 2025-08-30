#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backfill des métriques test pour ExtraTrees v2 à partir de predictions_test.csv.

Usage:
  python -m models.extratrees_plantvillage_v2.backfill_test_metrics

Le script lit:
  results/models/extratrees_plantvillage_v2/evaluation/predictions_test.csv
et écrit (ou réécrit):
  results/models/extratrees_plantvillage_v2/evaluation/test_metrics.json

Métriques calculées:
  - balanced_accuracy
  - f1_macro (métrique principale)
  - f1_micro
  - f1_weighted
  - accuracy
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

try:
    # Pour la cohérence des chemins du projet
    from src.helpers.helpers import PROJECT_ROOT
    PROJECT_ROOT = Path(PROJECT_ROOT)
except Exception:
    # Fallback: remonter à la racine du repo depuis ce fichier
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_DIR = PROJECT_ROOT / "results" / "models" / "extratrees_plantvillage_v2" / "evaluation"
PRED_PATH = EVAL_DIR / "predictions_test.csv"
OUT_PATH = EVAL_DIR / "test_metrics.json"


def main() -> int:
    if not PRED_PATH.exists():
        print(f"[ERROR] Fichier introuvable: {PRED_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(PRED_PATH)
    # Tolérer un éventuel index anonyme en première colonne
    if df.columns[0] not in {"y_true", "y_pred"}:
        # Essayer de drop la première colonne si elle ressemble à un index
        if df.columns[1:] .tolist() == ["y_true", "y_pred"]:
            df = df[["y_true", "y_pred"]]
        else:
            # Essayer de trouver les colonnes y_true/y_pred présentes
            cols = [c for c in df.columns if c in ("y_true", "y_pred")]
            if set(cols) == {"y_true", "y_pred"}:
                df = df[cols]
            else:
                print("[ERROR] Colonnes requises introuvables dans predictions_test.csv (attendu: y_true, y_pred)", file=sys.stderr)
                return 2

    y_true = df["y_true"].astype(str).values
    y_pred = df["y_pred"].astype(str).values

    # Calcul des métriques
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "accuracy": float(acc),
        "n_samples": int(len(y_true)),
        "n_classes": int(np.unique(y_true).shape[0]),
        "source": "backfill_from_predictions_test",
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("test_metrics.json écrit:")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
