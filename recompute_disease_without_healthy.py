import os
import sys
from typing import List, Dict, Tuple


def _parse_class_rows(report_path: str) -> List[Dict]:
    """Extrait les lignes de classes (label + precision/recall/f1/support)
    d'un classification_report sklearn en texte.

    On ignore volontairement les lignes globales 'accuracy', 'macro avg',
    'weighted avg' qui figurent en bas du rapport original.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        s = line.strip()
        if not s:
            continue

        tokens = s.split()
        # Lignes de classes: label (possiblement multi-mots) + 4 colonnes numériques
        if len(tokens) < 5:
            continue

        label = " ".join(tokens[:-4])
        label_lower = label.strip().lower()
        # Ignorer les lignes globales
        if label_lower in {"accuracy", "macro avg", "weighted avg"}:
            continue

        try:
            precision = float(tokens[-4])
            recall = float(tokens[-3])
            f1 = float(tokens[-2])
            support = int(tokens[-1])
        except ValueError:
            # Lignes type 'accuracy', 'macro avg', 'weighted avg' mal formées -> on ignore
            continue

        rows.append({
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    return rows


def _filter_disease_rows(rows: List[Dict]) -> List[Dict]:
    """Retourne les lignes de classes en excluant la classe 'healthy'."""
    disease_rows = [
        r for r in rows
        if r["label"].strip().lower() != "healthy"
    ]
    if not disease_rows:
        raise ValueError("Aucune classe maladie trouvée (ou pas de 'healthy').")
    return disease_rows


def _compute_global_metrics(disease_rows: List[Dict]) -> Tuple[float, float, float, float, float, float, float]:
    """Calcule accuracy, macro et weighted (precision/recall/F1) sur les seules maladies."""
    total_support = sum(r["support"] for r in disease_rows)
    # Accuracy = somme des TP / somme des supports (TP+FN)
    total_correct = sum(r["recall"] * r["support"] for r in disease_rows)
    accuracy = total_correct / total_support

    n_classes = len(disease_rows)
    macro_precision = sum(r["precision"] for r in disease_rows) / n_classes
    macro_recall = sum(r["recall"] for r in disease_rows) / n_classes
    macro_f1 = sum(r["f1"] for r in disease_rows) / n_classes

    # Weighted = moyenne pondérée par support
    weighted_precision = sum(r["precision"] * r["support"] for r in disease_rows) / total_support
    weighted_recall = sum(r["recall"] * r["support"] for r in disease_rows) / total_support
    weighted_f1 = sum(r["f1"] * r["support"] for r in disease_rows) / total_support

    return (
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1,
    )


def compute_metrics_without_healthy(report_path: str) -> Tuple[float, float]:
    """Recalcule accuracy et macro-F1 en excluant la classe 'healthy'."""
    rows = _parse_class_rows(report_path)
    disease_rows = _filter_disease_rows(rows)
    accuracy, _, _, macro_f1, _, _, _ = _compute_global_metrics(disease_rows)
    return accuracy, macro_f1


def write_report_without_healthy(report_path: str) -> str:
    """Crée un *nouveau* classification_report texte sans la classe 'healthy'.

    Le fichier original N'EST PAS modifié. On crée un fichier frère avec suffixe
    '_no_healthy.txt' dans le même dossier.
    """
    rows = _parse_class_rows(report_path)
    disease_rows = _filter_disease_rows(rows)
    (
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1,
    ) = _compute_global_metrics(disease_rows)

    # Préparer chemin de sortie
    dir_name, base_name = os.path.split(report_path)
    name, ext = os.path.splitext(base_name)
    out_name = f"{name}_no_healthy{ext or '.txt'}"
    out_path = os.path.join(dir_name, out_name)

    total_support = sum(r["support"] for r in disease_rows)

    # Construction d'un rapport texte simple (alignement type sklearn mais approximatif)
    lines = []
    lines.append("Classification report (diseases only, 'healthy' excluded)\n")
    lines.append(f"Source: {report_path}\n")
    lines.append("\n")
    lines.append("{:<40s} {:>9s} {:>9s} {:>9s} {:>9s}\n".format(
        "label", "precision", "recall", "f1-score", "support"
    ))
    lines.append("\n")
    for r in disease_rows:
        lines.append(
            "{:<40s} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}\n".format(
                r["label"], r["precision"], r["recall"], r["f1"], r["support"]
            )
        )

    lines.append("\n")
    # accuracy (une seule valeur)
    lines.append(
        "{:>40s} {:>9s} {:>9s} {:>9.4f} {:>9d}\n".format(
            "accuracy", "", "", accuracy, total_support
        )
    )
    # macro avg
    lines.append(
        "{:>40s} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}\n".format(
            "macro avg", macro_precision, macro_recall, macro_f1, total_support
        )
    )
    # weighted avg
    lines.append(
        "{:>40s} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}\n".format(
            "weighted avg", weighted_precision, weighted_recall, weighted_f1, total_support
        )
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return out_path


def main(paths):
    print("fichier\taccuracy_disease_only\tmacroF1_disease_only\trapport_sans_healthy")
    for p in paths:
        acc, mf1 = compute_metrics_without_healthy(p)
        out_path = write_report_without_healthy(p)
        print(f"{p}\t{acc:.4f}\t{mf1:.4f}\t{out_path}")


if __name__ == "__main__":
    # Soit chemins passés en arguments, soit quelques chemins codés en dur
    if len(sys.argv) > 1:
        report_paths = sys.argv[1:]
    else:
        report_paths = [
            "outputs_mono_1head_35classes/classification_report_disease_global.txt",
            "outputs_mono_disease_2heads_effv2s_256_color_split/classification_report.txt",
        ]
    main(report_paths)
