import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
import time
import joblib
import json
from sklearn.preprocessing import LabelEncoder

# Chemin racine du projet (1 niveau au-dessus de ce fichier : streamlit/app.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_clean_with_feature_csv import extract_all_features

# -------------------------
# Config
# -------------------------
st.set_page_config(
    page_title="Reconnaissance des plantes",
    page_icon="🌿",
    layout="wide"
)

# -------------------------
# State
# -------------------------
def init_state():
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = False
    if "step" not in st.session_state:
        st.session_state.step = 0

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_assets():
    """
    Chargement des assets : csv, images exemples, labels, etc.
    Retourne des objets Python (df, dict, listes...)
    """
    # Exemple :
    # df_features = pd.read_csv("data/features.csv")
    data_path = PROJECT_ROOT / "dataset/plantvillage/csv/fe_v1_clean_data_plantvillage_segmented_all_with_features.csv"
    df_features = None
    if data_path.exists():
        df_features = pd.read_csv(data_path)
    return df_features


@st.cache_data
def load_ml_artifacts():
    base = PROJECT_ROOT / "mlruns_MP_TESTS/207635128570842590/7b71c5048a294231a9fceac5b502176b"
    artifacts = {
        "metrics": {},
        "confusion_png": None,
        "per_class_f1_png": None,
        "classification_report": None,
    }

    if not base.exists():
        return artifacts

    metrics_dir = base / "metrics"
    for name in [
        "test_f1_macro",
        "test_balanced_accuracy",
        "cv_f1_macro_mean",
        "cv_bal_acc_mean",
    ]:
        path = metrics_dir / name
        if path.exists():
            try:
                value = float(path.read_text().strip())
                artifacts["metrics"][name] = value
            except Exception:
                continue

    cm_png = base / "artifacts/eval/confusion_matrix_norm.png"
    if cm_png.exists():
        artifacts["confusion_png"] = str(cm_png)

    f1_png = base / "artifacts/eval/per_class_f1.png"
    if f1_png.exists():
        artifacts["per_class_f1_png"] = str(f1_png)

    report_txt = base / "artifacts/eval/classification_report.txt"
    if report_txt.exists():
        try:
            artifacts["classification_report"] = report_txt.read_text(encoding="utf-8")
        except Exception:
            artifacts["classification_report"] = report_txt.read_text(errors="ignore")

    return artifacts


@st.cache_resource
def load_xgb_model():
    model_path = PROJECT_ROOT / "mlruns_MP_TESTS/207635128570842590/7b71c5048a294231a9fceac5b502176b/artifacts/model/model.pkl"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


@st.cache_data
def load_feature_names():
    feats_path = PROJECT_ROOT / "mlruns_MP_TESTS/207635128570842590/7b71c5048a294231a9fceac5b502176b/artifacts/data/features.json"
    if feats_path.exists():
        try:
            return json.loads(feats_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


@st.cache_data
def load_species_label_encoder():
    df = load_assets()
    if df is None or "nom_plante" not in df.columns:
        return None
    le = LabelEncoder()
    le.fit(df["nom_plante"].astype(str))
    return le


def build_features_from_image(pil_image: Image.Image) -> pd.DataFrame:
    """Extraire les features tabulaires à partir d'une image PIL.

    Utilise la même logique que le pipeline de feature engineering (224x224 + extract_all_features).
    """
    w, h = pil_image.size
    img_resized = pil_image.resize((224, 224))
    rgb = np.array(img_resized.convert("RGB"))
    feats = extract_all_features(rgb)

    feature_names = load_feature_names()
    if not feature_names:
        feature_names = sorted(feats.keys())

    # Compléter avec des métadonnées simples si attendues par le modèle
    if "width_img" in feature_names:
        feats["width_img"] = float(w)
    if "height_img" in feature_names:
        feats["height_img"] = float(h)

    row = {name: float(feats.get(name, 0.0)) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def header(title, subtitle=None):
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.title(title)
        if subtitle:
            st.caption(subtitle)
    with col2:
        st.markdown("")


# -------------------------
# Pages
# -------------------------
def page_home():
    header("🌿 Reconnaissance des plantes", "Démo Streamlit pour la soutenance")

    st.markdown("""
**Objectif :** à partir d’une photo de feuille :
- Identifier l’espèce
- Déterminer si la plante est saine ou malade
- Identifier la maladie si nécessaire
""")

    c1, c2, c3 = st.columns(3)
    c1.info("📌 **Contexte** : diagnostic rapide & assistance terrain")
    c2.info("🧠 **IA** : ML (features) vs DL (CNN / transfert)")
    c3.info("🚀 **Déploiement** : app interactive → usage réel")

    st.markdown("---")
    st.subheader("🎯 Message clé")
    st.write("Dataset contrôlé (type PlantVillage) → très bon score, mais attention à la généralisation en conditions réelles.")


def page_eda(df_features):
    header("📊 Exploration des données", "Comprendre le dataset et ses biais")

    if df_features is None or len(df_features) == 0:
        st.warning("Dataset PlantVillage non chargé. Vérifie le chemin du CSV dans load_assets().")
        return

    df = df_features.copy()
    if "nom_plante" not in df.columns or "nom_maladie" not in df.columns:
        st.warning("Le CSV chargé ne contient pas les colonnes attendues 'nom_plante' et 'nom_maladie'.")
        return

    df_classes = df.assign(
        label=lambda d: d["nom_plante"].astype(str) + "___" + d["nom_maladie"].astype(str)
    )

    n_images = len(df)
    n_species = df["nom_plante"].nunique()
    n_diseases = df["nom_maladie"].nunique()
    n_classes = df_classes["label"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", f"{n_images:,}".replace(",", " "))
    c2.metric("Espèces", int(n_species))
    c3.metric("Maladies", int(n_diseases))
    c4.metric("Classes (espèce + maladie)", int(n_classes))

    st.markdown("---")

    st.subheader("Explorer le dataset")
    objective = st.radio(
        "Angle d’analyse",
        ["Répartition des classes", "Sain vs Malade", "Exemples d’images"],
        horizontal=True,
    )

    if objective == "Répartition des classes":
        st.caption("Distribution des classes PlantVillage (espèce + maladie).")

        species_filter = st.multiselect(
            "Filtrer par espèce",
            options=sorted(df["nom_plante"].unique().tolist()),
            default=[],
        )

        df_counts = df_classes.groupby(["nom_plante", "label"]).size().reset_index(name="count")

        if species_filter:
            df_counts = df_counts[df_counts["nom_plante"].isin(species_filter)]

        if df_counts.empty:
            st.info("Aucune donnée pour le filtre sélectionné.")
        else:
            height = min(700, 20 * len(df_counts))
            chart = (
                alt.Chart(df_counts)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Nombre d'images"),
                    y=alt.Y("label:N", sort="-x", title="Classe"),
                    color=alt.Color("nom_plante:N", title="Espèce"),
                    tooltip=["label", "count"],
                )
                .properties(height=height)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

    elif objective == "Sain vs Malade":
        st.caption("Répartition des étiquettes saines / malades.")

        if "Est_Saine" not in df.columns:
            st.warning("La colonne 'Est_Saine' est absente du CSV.")
        else:
            df_health = df.copy()
            df_health["etat"] = np.where(df_health["Est_Saine"] == 1, "Saine", "Malade")
            counts = df_health["etat"].value_counts().reset_index()
            counts.columns = ["etat", "count"]

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(counts, use_container_width=True)
            with col2:
                chart = (
                    alt.Chart(counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("etat:N", title="État"),
                        y=alt.Y("count:Q", title="Nombre d'images"),
                        color="etat:N",
                        tooltip=["etat", "count"],
                    )
                )
                st.altair_chart(chart, use_container_width=True)

    else:
        st.caption("Échantillon d'images pour une classe PlantVillage (jeu réduit 5 images par classe).")
        root = PROJECT_ROOT / "dataset/plantvillage/data/plantvillage_5images"
        if not root.exists():
            st.warning("Le répertoire d'images d'exemple n'est pas disponible.")
        else:
            species = st.selectbox("Espèce", sorted(df["nom_plante"].unique().tolist()))
            subset = df[df["nom_plante"] == species]
            maladies = sorted(subset["nom_maladie"].unique().tolist())
            maladie = st.selectbox("Maladie / statut", maladies)

            folder_name = f"{species}___{maladie}"
            class_dir = root / folder_name

            if not class_dir.exists():
                st.warning(f"Dossier d'images introuvable pour {folder_name}.")
            else:
                exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
                image_paths = []
                for pattern in exts:
                    image_paths.extend(class_dir.glob(pattern))
                image_paths = sorted(image_paths)

                if not image_paths:
                    st.warning("Aucune image trouvée dans ce dossier.")
                else:
                    max_n = min(len(image_paths), 5)
                    n_show = st.slider("Nombre d'images à afficher", 1, max_n, max_n)
                    selected = image_paths[:n_show]
                    cols = st.columns(min(5, n_show))
                    for img_path, col in zip(selected, cols):
                        with col:
                            st.image(str(img_path), caption=img_path.name, use_container_width=True)

    st.warning("⚠️ Dataset PlantVillage en conditions contrôlées (fond uniforme, éclairage stable) → risque de baisse de performance sur photos terrain.")


def page_modeling():
    header("⚙️ Modélisation ML vs DL", "Choix méthodologiques et résultats")

    objective = st.radio(
        "Objectif",
        ["Objectif 1 — Espèce", "Objectif 2 — Santé", "Objectif 3 — Maladie"],
        horizontal=True
    )

    case = st.selectbox(
        "Cas d’usage (scénario produit)",
        [
            "Cas 1 — Identifier l’espèce",
            "Cas 2 — Espèce connue → diagnostiquer la maladie",
            "Cas 3 — Diagnostic complet"
        ]
    )

    c1, c2, c3 = st.columns(3)
    if objective.startswith("Objectif 1"):
        c1.metric("Type", "Multi-classe")
        c2.metric("Classes", "14 espèces")
        c3.metric("Focus", "Macro-F1 / confusion")
    elif objective.startswith("Objectif 2"):
        c1.metric("Type", "Binaire")
        c2.metric("Classes", "2 (healthy / diseased)")
        c3.metric("Focus", "Recall / F1")
    else:
        c1.metric("Type", "Multi-classe")
        c2.metric("Classes", "20 maladies")
        c3.metric("Risque", "déséquilibre / confusion")

    st.markdown("---")

    mode = st.radio(
        "Approche",
        ["Machine Learning (features engineering)", "Deep Learning (transfer learning)"],
        horizontal=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ML (features)")
        st.caption("Pipeline : features → modèle (LogReg / ExtraTrees / XGBoost / SVM-RBF).")
        st.progress(0.92)  # visuel
        st.markdown("""
- ✅ Rapide à entraîner
- ✅ Interprétable (jusqu’à un certain point)
- ⚠️ Dépend fortement de la qualité des features
""")

    with col2:
        st.subheader("DL (CNN / transfert)")
        st.caption("Pipeline : images → backbone pré-entraîné → fine-tuning.")
        st.progress(0.99)  # visuel
        st.markdown("""
- ✅ Très performant sur images
- ✅ Moins besoin de features manuelles
- ⚠️ Peut apprendre des biais (fond, conditions studio)
""")

    st.info(f"📌 Cas sélectionné : **{case}** • Approche affichée : **{mode}**")
    st.markdown("---")

    st.subheader("Résultats ML tabulaire (features PlantVillage)")
    artifacts = load_ml_artifacts()
    metrics = artifacts.get("metrics", {})

    col_a, col_b, col_c = st.columns(3)
    if metrics:
        test_f1 = metrics.get("test_f1_macro")
        test_bal = metrics.get("test_balanced_accuracy")
        cv_f1 = metrics.get("cv_f1_macro_mean")
        if test_f1 is not None:
            col_a.metric("F1 macro (test)", f"{test_f1:.3f}")
        else:
            col_a.metric("F1 macro (test)", "–")
        if test_bal is not None:
            col_b.metric("Balanced acc. (test)", f"{test_bal:.3f}")
        else:
            col_b.metric("Balanced acc. (test)", "–")
        if cv_f1 is not None:
            col_c.metric("F1 macro (CV moy.)", f"{cv_f1:.3f}")
        else:
            col_c.metric("F1 macro (CV moy.)", "–")
    else:
        col_a.metric("F1 macro (test)", "–")
        col_b.metric("Balanced acc. (test)", "–")
        col_c.metric("F1 macro (CV moy.)", "–")

    st.caption("Modèle affiché : XGBoost (meilleur run MLflow sur PlantVillage tabulaire).")

    tab_cm, tab_f1, tab_rep = st.tabs([
        "Matrice de confusion",
        "F1 par classe",
        "Rapport de classification",
    ])

    with tab_cm:
        if artifacts.get("confusion_png"):
            st.image(
                artifacts["confusion_png"],
                caption="Matrice de confusion normalisée (PlantVillage)",
                use_container_width=True,
            )
        else:
            st.info("Image de matrice de confusion non trouvée dans les artefacts MLflow.")

    with tab_f1:
        if artifacts.get("per_class_f1_png"):
            st.image(
                artifacts["per_class_f1_png"],
                caption="F1-score par classe (PlantVillage)",
                use_container_width=True,
            )
        else:
            st.info("Figure de F1 par classe non trouvée dans les artefacts MLflow.")

    with tab_rep:
        report = artifacts.get("classification_report")
        if report:
            st.text(report)
        else:
            st.info("Fichier classification_report.txt non trouvé dans les artefacts MLflow.")


def page_predict():
    header("🤖 Démo de prédiction", "Upload → prédiction → top-3 → décision")

    model = load_xgb_model()
    label_encoder = load_species_label_encoder()

    if model is None or label_encoder is None:
        st.error("Modèle XGBoost ou label encoder introuvable. Vérifie les artefacts MLflow et le CSV.")
        return

    st.subheader("Réglages (dynamique)")
    threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.70, 0.01)

    bg = st.selectbox("What-if : fond (démo robustesse)", ["Aucun", "Noir", "Blanc", "Saumon", "Vert"])
    bg_map = {
        "Noir": (0, 0, 0),
        "Blanc": (255, 255, 255),
        "Saumon": (250, 128, 114),
        "Vert": (0, 140, 60),
    }

    st.markdown("---")

    uploaded = st.file_uploader("Uploader une image de feuille", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Charge une image pour lancer la démo.")
        return

    img = Image.open(uploaded).convert("RGB")

    # Simulation changement de fond (visuel “wow”)
    if bg != "Aucun":
        w, h = img.size
        background = Image.new("RGB", (w, h), bg_map[bg])
        alpha = st.slider("Intensité effet fond (démo)", 0.0, 1.0, 0.25, 0.05)
        img_to_show = Image.blend(img, background, alpha=alpha)
    else:
        img_to_show = img

    col1, col2 = st.columns([0.55, 0.45])
    with col1:
        st.image(img_to_show, caption="Image (éventuellement modifiée)", use_container_width=True)

    with col2:
        st.subheader("Résultats")
        st.caption("Modèle : XGBoost tabulaire (classification d'espèce à partir de features).")

        if st.button("Prédire"):
            with st.spinner("Extraction des features (tabulaires)…"):
                X = build_features_from_image(img_to_show)

            try:
                with st.spinner("Inférence modèle XGBoost…"):
                    proba = model.predict_proba(X)[0]
                    class_indices = np.argsort(proba)[::-1]
                    classes_encoded = np.asarray(model.classes_).astype(int)
            except Exception as e:
                st.error(f"Erreur pendant l'inférence : {e}")
                return

            k = min(3, len(class_indices))
            idx_topk = class_indices[:k]
            species_ids = classes_encoded[idx_topk]
            labels = label_encoder.inverse_transform(species_ids)
            topk = list(zip(labels, proba[idx_topk]))

            st.markdown("### Top-3 prédictions (espèce)")
            for label, p in topk:
                st.write(f"**{label}** — {p:.2f}")
                st.progress(float(p))

            best_label, best_p = topk[0]
            st.markdown("---")
            if best_p < threshold:
                st.warning("Confiance insuffisante → demander une autre photo / zoom / segmentation.")
            else:
                st.success(f"Décision : **{best_label}** (p={best_p:.2f})")


def page_gradcam():
    header("🔍 Interprétabilité (Grad-CAM)", "Expliquer pourquoi le modèle prédit ça")

    st.markdown("""
Ici, tu vas montrer :
- image originale
- heatmap (Grad-CAM)
- commentaire : “le modèle regarde bien la feuille” ou “biais fond”
""")

    show = st.checkbox("Afficher des exemples Grad-CAM", value=False)
    if not show:
        return

    root = PROJECT_ROOT / "reports/gradcam"
    if not root.exists():
        st.info("Mettre des exemples dans `reports/gradcam/<exemple>/original.png` et `reports/gradcam/<exemple>/gradcam.png` (copiés depuis ton Drive).")
        return

    example_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not example_dirs:
        st.info("Aucun sous-dossier trouvé dans `reports/gradcam`. Crée par exemple `reports/gradcam/tomate_bien_predite/` avec `original.png` et `gradcam.png`.")
        return

    for ex_dir in example_dirs:
        orig = ex_dir / "original.png"
        heat = ex_dir / "gradcam.png"
        if not (orig.exists() and heat.exists()):
            continue

        st.markdown(f"### Exemple : {ex_dir.name}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Image originale**")
            st.image(str(orig), use_container_width=True)
        with col2:
            st.write("**Heatmap Grad-CAM**")
            st.image(str(heat), use_container_width=True)


def page_limits():
    header("⚠️ Limites & perspectives", "Posture réaliste et axes d’amélioration")

    st.subheader("Limites")
    st.markdown("""
- Dataset très “studio” → généralisation limitée sur photos terrain
- Déséquilibre des classes → biais possible
- Fond / éclairage peuvent influencer la prédiction
""")

    st.subheader("Perspectives")
    st.markdown("""
- Ajouter des données “in the wild” (terrain)
- Segmentation pour isoler la feuille
- Augmentations plus réalistes (fonds variés, illumination)
- Tester d’autres backbones / ViT + validation plus robuste
""")


def page_about():
    header("ℹ️ À propos")
    st.markdown("""
- Projet : Reconnaissance plantes (espèce / santé / maladie)
- Stack : Python • scikit-learn • TensorFlow/Keras • Streamlit
- Objectif soutenance : démontrer pipeline + choix + limites + démo produit
""")


# -------------------------
# App Router
# -------------------------
def main():
    df_features = load_assets()
    init_state()

    st.sidebar.title("Navigation")
    st.sidebar.toggle("🎬 Mode soutenance (démo guidée)", key="demo_mode")
    st.sidebar.markdown("---")

    pages = [
        "🏠 Accueil",
        "📊 Exploration des données",
        "⚙️ Modélisation ML vs DL",
        "🤖 Démo de prédiction",
        "🔍 Interprétabilité (Grad-CAM)",
        "⚠️ Limites & perspectives",
        "ℹ️ À propos"
    ]

    if st.session_state.demo_mode:
        colA, colB = st.sidebar.columns([0.7, 0.3])
        with colA:
            st.sidebar.write(f"Étape : **{pages[st.session_state.step]}**")
        with colB:
            if st.sidebar.button("Next ➡️"):
                st.session_state.step = (st.session_state.step + 1) % len(pages)

        st.sidebar.progress((st.session_state.step + 1) / len(pages))
        page = pages[st.session_state.step]
    else:
        page = st.sidebar.radio("Menu", pages)

    # Routing
    if page == "🏠 Accueil":
        page_home()
    elif page == "📊 Exploration des données":
        page_eda(df_features)
    elif page == "⚙️ Modélisation ML vs DL":
        page_modeling()
    elif page == "🤖 Démo de prédiction":
        page_predict()
    elif page == "🔍 Interprétabilité (Grad-CAM)":
        page_gradcam()
    elif page == "⚠️ Limites & perspectives":
        page_limits()
    else:
        page_about()


if __name__ == "__main__":
    main()
