
"""
train_svm_from_embeddings.py
----------------------------
Trains SVMs on Keras backbone embeddings exported by export_embeddings_keras.py.

Models trained:
  - Species SVM (multi-class, LinearSVC and RBF SVC via grid)
  - Health SVM (binary, class_weight="balanced")
  - Disease SVM:
       Option A (per-species) : one SVM per species (multi-class diseases for that species)
       Option B (global)      : one global SVM (multi-class) trained only on diseased samples

Outputs:
 - metrics.json with accuracies & macro-F1
 - confusion matrices as NPY/CSV for species, health, and disease (global and per species)
 - trained scikit-learn models as joblib files

Usage:
  python train_svm_from_embeddings.py --emb_dir outputs_embeddings --outdir svm_results --pca_dim 256 --grid

Requires: scikit-learn, numpy, joblib

But : entraîner des SVM (modèles scikit-learn) sur les embeddings (vecteurs de caractéristiques) extraits par ton backbone Keras, 
puis évaluer et sauvegarder des métriques + matrices de confusion.
Entrées attendues (dans --emb_dir) :
X_train.npy, X_val.npy, X_test.npy : matrices (N, D) (embeddings).
y_species_*.npy : labels d’espèces (N,) (entiers).
y_health_*.npy : labels santé (N,) (0 = healthy, 1 = diseased).
y_disease_*.npy : labels maladie (N,) (entiers pour malades, -1 si healthy).
Sorties (dans --outdir) : metrics.json, matrices .npy/.csv, et modèles SVM .joblib.
"""

import os, json, argparse
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Noms lisibles (chargés depuis emb_dir/species.json et emb_dir/diseases.json dans main)
species_names = None
disease_names = None
# Default globals (overridden in main)
pca_dim = 256
n_jobs = -1
cv_folds = 3
# Scoring strategies (overridden in main)
scoring_species = "accuracy"
scoring_health = "accuracy"
scoring_disease = "f1_macro"

"""
parse_argsRôle. Lire les arguments de la ligne de commande.
Entrées. —
Sorties. Un objet args avec :
emb_dir : dossier où se trouvent les .npy exportés ;
outdir : où sauver résultats et modèles ;
pca_dim : dimension PCA (0 = sans PCA) ;
grid : booléen pour activer une petite grid search (SVM RBF).
Détails. Utilise argparse pour rendre le script configurable depuis le terminal.
"""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True, type=str, help="Directory with X_*.npy and labels from export script")
    ap.add_argument("--outdir", type=str, default="svm_results")
    ap.add_argument("--pca_dim", type=int, default=256, help="0 to disable PCA")
    ap.add_argument("--grid", action="store_true", help="Run small grid search for RBF C/gamma")
    ap.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV (-1: all cores)")
    ap.add_argument("--cv", type=int, default=3, help="Number of CV folds for GridSearchCV")
    ap.add_argument("--scoring", type=str, default="auto",
                    help="Scoring for GridSearchCV. 'auto' => species/health: accuracy; disease: f1_macro. Or choose: accuracy, f1_macro, balanced_accuracy, f1_weighted")
    ap.add_argument("--scoring_species", type=str, default=None, help="Override scoring for species head (else use --scoring)")
    ap.add_argument("--scoring_health", type=str, default=None, help="Override scoring for health head (else use --scoring)")
    ap.add_argument("--scoring_disease", type=str, default=None, help="Override scoring for disease heads (global and per species)")
    return ap.parse_args()

"""
Rôle. Charger en mémoire les embeddings et labels pour un split donné ("train", "val", "test").
Entrées.
emb_dir : dossier des .npy ;
split : "train", "val" ou "test".
Sorties.
X : np.ndarray (N, D) — embeddings ;
ysp : np.ndarray (N,) — labels species ;
yhl : np.ndarray (N,) — labels health (0/1) ;
ydi : np.ndarray (N,) — labels disease (entier si malade, -1 si healthy).
Détails. Les noms de fichiers suivent la convention déposée par le script d’export.
"""

def load_split(emb_dir, split):
    X = np.load(os.path.join(emb_dir, f"X_{split}.npy"))
    ysp = np.load(os.path.join(emb_dir, f"y_species_{split}.npy"))
    yhl = np.load(os.path.join(emb_dir, f"y_health_{split}.npy"))
    ydi = np.load(os.path.join(emb_dir, f"y_disease_{split}.npy"))
    return X, ysp, yhl, ydi

"""
pipeline(pca_dim):
Rôle. Construire un pré-traitement scikit-learn standard pour les SVM.
Entrées.
pca_dim : dimension cible pour la PCA (réduction dimensionnelle). 0 => pas de PCA.
Sorties. Une liste d’étapes (pas encore le classifieur) :
StandardScaler() : centre et met à l’échelle les features ;
PCA(n_components=pca_dim) (optionnel) : compresse les embeddings (utile pour SVM RBF / vitesse).
Détails. On ajoutera ensuite la dernière étape ("clf", SVM) pour créer un Pipeline.
"""

def pipeline(pca_dim):
    steps = [("scaler", StandardScaler())]
    if pca_dim and pca_dim>0:
        steps.append(("pca", PCA(n_components=pca_dim, whiten=False, random_state=42)))
    return steps

"""
save_classification_report(y_true, y_pred, outdir, base_name, labels=None, target_names=None, header=None)
Rôle. Sauvegarder un classification_report scikit-learn en .txt (lisible) et .json.
Détails.
- labels: ordre des classes (par défaut: classes présentes dans y_true triées).
- target_names: noms lisibles des classes (même longueur que labels) ; si None, garde les ids.
- header: texte facultatif ajouté en tête du .txt (ex: espèce concernée).
"""

def save_classification_report(y_true, y_pred, outdir, base_name, labels=None, target_names=None, header=None):
    os.makedirs(outdir, exist_ok=True)
    if labels is None:
        labels = sorted(list({int(x) for x in np.unique(y_true)}))
    # Aligner target_names avec labels si fournis
    kwargs = {"labels": labels, "digits": 4, "zero_division": 0}
    if target_names is not None and len(target_names) == len(labels):
        kwargs["target_names"] = target_names
    # Texte lisible
    report_txt = classification_report(y_true, y_pred, **kwargs)
    # Version JSON structurée
    report_json = classification_report(y_true, y_pred, output_dict=True, **kwargs)
    # Écritures
    txt_path = os.path.join(outdir, f"{base_name}.txt")
    json_path = os.path.join(outdir, f"{base_name}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip("\n") + "\n\n")
        f.write(report_txt + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

def _json_safe(obj):
    """
    Rendre un objet JSON-sérialisable (convertit numpy.* et ndarrays en types Python).
    """
    import numpy as _np
    import numpy.ma as _ma
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if _ma.isMaskedArray(obj):
        # Remplacer les valeurs masquées par None (sûr pour JSON), sans imposer NaN sur des dtypes entiers
        data = obj.data
        mask = obj.mask
        if mask is _ma.nomask:
            return _json_safe(data)
        data_obj = data.astype(object)
        data_obj[mask] = None
        return _json_safe(data_obj.tolist())
    if isinstance(obj, _np.ndarray):
        # Convertir en liste puis appliquer récursivement (pour objets imbriqués)
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, _np.generic):
        return obj.item()
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def save_cv_results_if_any(clf, outdir, filename="cv_results.json"):
    """
    Si clf est un GridSearchCV, sauve best_params_, best_score_ et cv_results_ en JSON.
    """
    if hasattr(clf, "cv_results_"):
        payload = {
            "best_params": getattr(clf, "best_params_", None),
            "best_score": float(getattr(clf, "best_score_", float("nan"))),
            "scorer": str(getattr(clf, "scorer_", "unknown")),
            "cv_results": getattr(clf, "cv_results_", {}),
        }
        payload = _json_safe(payload)
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

"""
train_eval_species(Xtr, ytr, Xte, yte, outdir, grid=False):
Rôle. Entraîner un SVM pour l’espèce (multi-classe) et évaluer sur test.
Entrées.
Xtr, ytr : embeddings + labels espèce (train, ici train+val dans le main) ;
Xte, yte : embeddings + labels espèce (test) ;
outdir : dossier pour sauver résultats de cette tâche ;
grid : si True, fait une grid search SVM RBF (C, gamma) ; sinon LinearSVC (rapide, robuste).
Sorties.
metrics dict : {"accuracy": ..., "macroF1": ...}.
Étapes.
Construit le pipeline : Scaler (+ PCA) + classifieur (LinearSVC par défaut, ou SVC RBF si grid).
Fit sur Xtr, ytr.
Predict sur Xte.
Calcule matrice de confusion, accuracy, macro-F1.
Sauvegarde cm_species.npy/.csv et le modèle svm_species.joblib.
Retourne le dictionnaire de métriques.
Pourquoi macro-F1 ? Pour pondérer équitablement les classes, même si elles sont déséquilibrées.
"""

def train_eval_species(Xtr, ytr, Xte, yte, outdir, grid=False):
    os.makedirs(outdir, exist_ok=True)
    steps = pipeline(pca_dim)
    if grid:
        svc = SVC(kernel="rbf", class_weight="balanced")
        model = Pipeline(steps + [("clf", svc)])
        param_grid = {
            "scaler": [StandardScaler(), RobustScaler()],
            "clf__C": [1, 10, 100],
            "clf__gamma": ["scale", 1e-3, 1e-4],
        }
        clf = GridSearchCV(model, param_grid, cv=cv_folds, n_jobs=n_jobs, verbose=1, scoring=scoring_species)
    else:
        clf = Pipeline(steps + [("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto"))])
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    cm = confusion_matrix(yte, ypred)
    metrics = {"accuracy": float(accuracy_score(yte, ypred)), "macroF1": float(f1_score(yte, ypred, average="macro"))}
    np.save(os.path.join(outdir, "cm_species.npy"), cm)
    np.savetxt(os.path.join(outdir, "cm_species.csv"), cm, fmt="%d", delimiter=",")
    joblib.dump(clf, os.path.join(outdir, "svm_species.joblib"))
    # Audit surapprentissage: évaluer sur TRAIN (train+val)
    ypred_tr = clf.predict(Xtr)
    cm_tr = confusion_matrix(ytr, ypred_tr)
    np.save(os.path.join(outdir, "cm_species_train.npy"), cm_tr)
    np.savetxt(os.path.join(outdir, "cm_species_train.csv"), cm_tr, fmt="%d", delimiter=",")
    labels_tr = sorted(list({int(x) for x in np.unique(ytr)}))
    tnames_tr = [species_names[i] for i in labels_tr] if species_names is not None else None
    save_classification_report(ytr, ypred_tr, outdir, "classification_report_species_train", labels=labels_tr, target_names=tnames_tr)
    # Sauvegarde CV si grid
    save_cv_results_if_any(clf, outdir, filename="cv_results.json")
    # Classification report (avec noms d'espèces si disponibles)
    labels = sorted(list({int(x) for x in np.unique(yte)}))
    tnames = [species_names[i] for i in labels] if species_names is not None else None
    save_classification_report(yte, ypred, outdir, "classification_report_species", labels=labels, target_names=tnames)
    return metrics

"""
train_eval_health(Xtr, ytr, Xte, yte, outdir, grid=False):
Rôle. Entraîner un SVM binaire (sain vs malade) et évaluer.
Entrées/Sorties. Idem train_eval_species, mais avec labels health (0/1).
Spécificités.
class_weight="balanced" aide si la proportion sain/malade est déséquilibrée.
macro-F1 te donne une idée du rappel sur la classe minoritaire (souvent “malade”).
"""

def train_eval_health(Xtr, ytr, Xte, yte, outdir, grid=False):
    os.makedirs(outdir, exist_ok=True)
    steps = pipeline(pca_dim)
    if grid:
        svc = SVC(kernel="rbf", class_weight="balanced", probability=True)
        model = Pipeline(steps + [("clf", svc)])
        param_grid = {
            "scaler": [StandardScaler(), RobustScaler()],
            "clf__C": [0.5, 1, 2],
            "clf__gamma": ["scale", 1e-3],
        }
        clf = GridSearchCV(model, param_grid, cv=cv_folds, n_jobs=n_jobs, verbose=1, scoring=scoring_health)
    else:
        clf = Pipeline(steps + [("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto"))])
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    cm = confusion_matrix(yte, ypred)
    metrics = {"accuracy": float(accuracy_score(yte, ypred)), "macroF1": float(f1_score(yte, ypred, average="macro"))}
    np.save(os.path.join(outdir, "cm_health.npy"), cm)
    np.savetxt(os.path.join(outdir, "cm_health.csv"), cm, fmt="%d", delimiter=",")
    joblib.dump(clf, os.path.join(outdir, "svm_health.joblib"))
    # Audit surapprentissage: TRAIN (train+val)
    ypred_tr = clf.predict(Xtr)
    cm_tr = confusion_matrix(ytr, ypred_tr)
    np.save(os.path.join(outdir, "cm_health_train.npy"), cm_tr)
    np.savetxt(os.path.join(outdir, "cm_health_train.csv"), cm_tr, fmt="%d", delimiter=",")
    save_classification_report(ytr, ypred_tr, outdir, "classification_report_health_train", labels=[0,1], target_names=["healthy","diseased"])
    # Sauvegarde CV si grid
    save_cv_results_if_any(clf, outdir, filename="cv_results.json")
    # Classification report binaire
    labels = [0, 1]
    tnames = ["healthy", "diseased"]
    save_classification_report(yte, ypred, outdir, "classification_report_health", labels=labels, target_names=tnames)
    return metrics

"""
train_eval_disease_global(Xtr, ytr, Xte, yte, outdir, grid=False):
Rôle. Entraîner un SVM multi-classe pour la maladie, globalement (toutes espèces confondues) — uniquement sur les échantillons malades.
Entrées.
Xtr, ytr : embeddings + labels disease (train+val) ;
Xte, yte : embeddings + labels disease (test) ;
outdir, grid : comme plus haut.
Sorties.
metrics dict : {"accuracy": ..., "macroF1": ...}.
Étapes.
Filtre les lignes où y != -1 (on retire les healthy).
Construit le pipeline (Scaler (+ PCA) + SVM).
Fit → Predict → confusion matrix → accuracy + macro-F1.
Sauvegarde cm_disease_global.npy/.csv et svm_disease_global.joblib.
Pourquoi filtrer ? Le label maladie n’existe que pour les feuilles malades ; -1 est un marqueur “pas de maladie”.
"""

def train_eval_disease_global(Xtr, ytr, Xte, yte, outdir, grid=False):
    tr_idx = ytr!=-1
    te_idx = yte!=-1
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]
    Xte, yte = Xte[te_idx], yte[te_idx]
    os.makedirs(outdir, exist_ok=True)
    steps = pipeline(pca_dim)
    if grid:
        svc = SVC(kernel="rbf", class_weight="balanced", probability=True)
        model = Pipeline(steps + [("clf", svc)])
        param_grid = {
            "scaler": [StandardScaler(), RobustScaler()],
            "clf__C": [0.5, 1, 2, 10, 100],
            "clf__gamma": ["scale", 1e-3, 1e-4],
        }
        clf = GridSearchCV(model, param_grid, cv=cv_folds, n_jobs=n_jobs, verbose=1, scoring=scoring_disease)
    else:
        clf = Pipeline(steps + [("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto"))])
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    cm = confusion_matrix(yte, ypred)
    metrics = {"accuracy": float(accuracy_score(yte, ypred)), "macroF1": float(f1_score(yte, ypred, average="macro"))}
    np.save(os.path.join(outdir, "cm_disease_global.npy"), cm)
    np.savetxt(os.path.join(outdir, "cm_disease_global.csv"), cm, fmt="%d", delimiter=",")
    joblib.dump(clf, os.path.join(outdir, "svm_disease_global.joblib"))
    # Audit surapprentissage: TRAIN (malades uniquement)
    ypred_tr = clf.predict(Xtr)
    cm_tr = confusion_matrix(ytr, ypred_tr)
    np.save(os.path.join(outdir, "cm_disease_global_train.npy"), cm_tr)
    np.savetxt(os.path.join(outdir, "cm_disease_global_train.csv"), cm_tr, fmt="%d", delimiter=",")
    labels_tr = sorted(list({int(x) for x in np.unique(ytr)}))
    tnames_tr = [disease_names[i] for i in labels_tr] if disease_names is not None else None
    save_classification_report(ytr, ypred_tr, outdir, "classification_report_disease_global_train", labels=labels_tr, target_names=tnames_tr)
    # Sauvegarde CV si grid
    save_cv_results_if_any(clf, outdir, filename="cv_results.json")
    # Classification report maladies (global, hors healthy)
    labels = sorted(list({int(x) for x in np.unique(yte)}))
    tnames = [disease_names[i] for i in labels] if disease_names is not None else None
    save_classification_report(yte, ypred, outdir, "classification_report_disease_global", labels=labels, target_names=tnames)
    return metrics

"""
train_eval_disease_per_species(Xtr, ysp_tr, ydi_tr, Xte, ysp_te, ydi_te, outdir, grid=False):
Rôle. Entraîner un SVM par espèce pour classer les maladies de cette espèce (toujours sans healthy).
Entrées.
Xtr, ysp_tr, ydi_tr : embeddings + labels espèce + labels maladie (train+val) ;
Xte, ysp_te, ydi_te : idem pour test ;
outdir, grid.
Sorties.
metrics dict indexé par id d’espèce : {species_id: {"accuracy": ..., "macroF1": ...}, ...}.
Étapes.
Récupère la liste des espèces présentes.
Pour chaque espèce :
prend uniquement ses images malades (y_disease != -1) dans cette espèce ;
entraîne un SVM (Scaler (+ PCA) + classifieur) ;
évalue sur test filtré de la même façon ;
sauve la matrice de confusion et le modèle svm_disease_species_<id>.joblib.
Retourne le dict de métriques.
Intérêt. Réduit la confusion inter-espèces : le classifieur ne voit que les maladies pertinentes d’une espèce donnée.
"""

def train_eval_disease_per_species(Xtr, ysp_tr, ydi_tr, Xte, ysp_te, ydi_te, outdir, grid=False):
    os.makedirs(outdir, exist_ok=True)
    species_ids = np.unique(ysp_tr)
    metrics = {}
    skipped = {}

    for sp in species_ids:
        tr_idx = (ysp_tr == sp) & (ydi_tr != -1)
        te_idx = (ysp_te == sp) & (ydi_te != -1)

        if tr_idx.sum() == 0 or te_idx.sum() == 0:
            skipped[int(sp)] = "no diseased samples in train or test"
            continue

        Xt, yt = Xtr[tr_idx], ydi_tr[tr_idx]
        Xv, yv = Xte[te_idx], ydi_te[te_idx]

        # ⛔️ Besoin d'au moins 2 classes en TRAIN
        if np.unique(yt).size < 2:
            skipped[int(sp)] = "only one disease class in train"
            continue

        steps = pipeline(pca_dim)
        if grid:
            svc = SVC(kernel="rbf", class_weight="balanced", probability=True)
            model = Pipeline(steps + [("clf", svc)])
            param_grid = {
                "scaler": [StandardScaler(), RobustScaler()],
                "clf__C": [0.5, 1, 2, 10],
                "clf__gamma": ["scale", 1e-3, 1e-4],
            }
            clf = GridSearchCV(model, param_grid, cv=cv_folds, n_jobs=n_jobs, verbose=0, scoring=scoring_disease)
        else:
            clf = Pipeline(steps + [("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000))])

        clf.fit(Xt, yt)
        yp = clf.predict(Xv)

        cm = confusion_matrix(yv, yp)
        acc = float(accuracy_score(yv, yp))
        mf1 = float(f1_score(yv, yp, average="macro"))
        metrics[int(sp)] = {"accuracy": acc, "macroF1": mf1}

        np.save(os.path.join(outdir, f"cm_disease_species_{int(sp)}.npy"), cm)
        np.savetxt(os.path.join(outdir, f"cm_disease_species_{int(sp)}.csv"), cm, fmt="%d", delimiter=",")
        joblib.dump(clf, os.path.join(outdir, f"svm_disease_species_{int(sp)}.joblib"))
        # Audit TRAIN (malades uniquement pour cette espèce)
        yp_tr = clf.predict(Xt)
        cm_tr = confusion_matrix(yt, yp_tr)
        np.save(os.path.join(outdir, f"cm_disease_species_{int(sp)}_train.npy"), cm_tr)
        np.savetxt(os.path.join(outdir, f"cm_disease_species_{int(sp)}_train.csv"), cm_tr, fmt="%d", delimiter=",")
        labels_tr = sorted(list({int(x) for x in np.unique(yt)}))
        tnames_tr = [disease_names[i] for i in labels_tr] if disease_names is not None else None
        sp_name = species_names[int(sp)] if species_names is not None and int(sp) < len(species_names) else str(int(sp))
        header_tr = f"Species {int(sp)} — {sp_name} (TRAIN)"
        save_classification_report(yt, yp_tr, outdir, f"classification_report_disease_species_{int(sp)}_train", labels=labels_tr, target_names=tnames_tr, header=header_tr)
        # Sauvegarde CV si grid
        save_cv_results_if_any(clf, outdir, filename=f"cv_results_species_{int(sp)}.json")
        # Classification report par espèce (seulement les maladies présentes)
        labels = sorted(list({int(x) for x in np.unique(yv)}))
        tnames = [disease_names[i] for i in labels] if disease_names is not None else None
        sp_name = species_names[int(sp)] if species_names is not None and int(sp) < len(species_names) else str(int(sp))
        header = f"Species {int(sp)} — {sp_name}"
        save_classification_report(yv, yp, outdir, f"classification_report_disease_species_{int(sp)}", labels=labels, target_names=tnames, header=header)

    # journal des espèces ignorées
    with open(os.path.join(outdir, "skipped_species.json"), "w") as f:
        json.dump(skipped, f, indent=2)

    return metrics


"""
if __name__ == "__main__": (le “main”)
Rôle. Orchestration : charge les données, entraîne chaque SVM, agrège et sauve les résultats.
Étapes.
Lit les arguments (parse_args()), crée outdir.
Charge train/val/test via load_split(...).
Concatène train + val pour entraîner les SVM (pratique standard) ; garde test pour l’évaluation finale.
Appelle, l’un après l’autre :
train_eval_species(...) → results["species"]
train_eval_health(...) → results["health"]
train_eval_disease_global(...) → results["disease_global"]
train_eval_disease_per_species(...) → results["disease_per_species"]
Écrit toutes les métriques dans metrics.json.
Affiche les résultats (pratique pour copier-coller dans ton rapport).
"""

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pca_dim = args.pca_dim
    n_jobs = args.n_jobs
    cv_folds = args.cv
    # Resolve scoring strategies (auto or overrides)
    def _resolve_scoring(arg_value, default_value):
        if arg_value is None:
            return default_value
        val = str(arg_value).lower()
        if val == "auto":
            return default_value
        return arg_value

    default_species = "accuracy"
    default_health = "accuracy"
    default_disease = "f1_macro"
    scoring_species = _resolve_scoring(getattr(args, "scoring_species", None) or args.scoring, default_species)
    scoring_health = _resolve_scoring(getattr(args, "scoring_health", None) or args.scoring, default_health)
    scoring_disease = _resolve_scoring(getattr(args, "scoring_disease", None) or args.scoring, default_disease)
    # Charger les noms lisibles des classes (si disponibles)
    try:
        with open(os.path.join(args.emb_dir, "species.json"), "r", encoding="utf-8") as f:
            species_names = json.load(f)
    except Exception:
        species_names = None
    try:
        with open(os.path.join(args.emb_dir, "diseases.json"), "r", encoding="utf-8") as f:
            disease_names = json.load(f)
    except Exception:
        disease_names = None

    Xtr, ysp_tr, yhl_tr, ydi_tr = load_split(args.emb_dir, "train")
    Xva, ysp_va, yhl_va, ydi_va = load_split(args.emb_dir, "val")
    Xte, ysp_te, yhl_te, ydi_te = load_split(args.emb_dir, "test")

    # Merge train+val for final training, evaluate on test
    Xtr_full = np.concatenate([Xtr, Xva], axis=0)
    ysp_tr_full = np.concatenate([ysp_tr, ysp_va], axis=0)
    yhl_tr_full = np.concatenate([yhl_tr, yhl_va], axis=0)
    ydi_tr_full = np.concatenate([ydi_tr, ydi_va], axis=0)

    results = {}

    results["species"] = train_eval_species(
        Xtr_full, ysp_tr_full, Xte, ysp_te, os.path.join(args.outdir, "species"), grid=args.grid
    )
    results["health"] = train_eval_health(
        Xtr_full, yhl_tr_full, Xte, yhl_te, os.path.join(args.outdir, "health"), grid=args.grid
    )
    results["disease_global"] = train_eval_disease_global(
        Xtr_full, ydi_tr_full, Xte, ydi_te, os.path.join(args.outdir, "disease_global"), grid=args.grid
    )
    results["disease_per_species"] = train_eval_disease_per_species(
    Xtr_full, ysp_tr_full, ydi_tr_full, Xte, ysp_te, ydi_te,
    os.path.join(args.outdir, "disease_per_species"), grid=args.grid
    )

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
