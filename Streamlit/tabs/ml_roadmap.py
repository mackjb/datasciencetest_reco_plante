import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Helper functions for reporting (reused)
def parse_classification_report(file_path):
    try:
        with open(file_path, "r") as f:
            report_text = f.read()
        lines = [line.strip() for line in report_text.split('\n') if line.strip()]
        data = []
        for line in lines[1:]:
             parts = line.split()
             if len(parts) >= 2:
                if parts[0] in ['accuracy', 'macro', 'weighted']: continue
                name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1 = float(parts[3])
                support = int(parts[4])
                data.append([name, precision, recall, f1, support])
        df = pd.DataFrame(data, columns=["Classe", "Précision", "Rappel", "F1-score", "Support"])
        return df
    except Exception as e:
        return None

def render_df_as_image(df, title=None):
    try:
        fig, ax = plt.subplots(figsize=(5, len(df) * 0.25 + 1))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        if title: plt.title(title, fontweight="bold")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None

def render_roadmap():
    if 'ml_step' not in st.session_state:
        st.session_state['ml_step'] = 1

    steps = {
        1: "Collecte",
        2: "Extraction",
        3: "Nettoyage",
        4: "Scaling",
        5: "Modélisation"
    }
    
    # Custom CSS to make arrows vertical-align
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    
    /* Style uniquement pour le bouton ACTIF (Primary) */
    div.stButton > button[kind="primary"] {
        background-image: linear-gradient(#2e7d32, #1b5e20) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    /* Style pour les boutons INACTIFS (Secondary) */
    div.stButton > button[kind="secondary"] {
        background-color: #f0f2f6 !important;
        color: #31333F !important;
        border: 1px solid #d6d6d6 !important;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)


    
    # Create a dynamic layout: Button -> Arrow -> Button ...
    # 5 steps + 4 arrows = 9 columns
    # We tweak ratios: Buttons (2), Arrows (0.5)
    cols = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2])
    
    for idx, (step_id, step_label) in enumerate(steps.items()):
        # Calculate column index for button: 0, 2, 4, 6, 8
        col_idx = idx * 2
        
        with cols[col_idx]:
            is_active = (st.session_state['ml_step'] == step_id)
            # Use emoji in label for flair
            label_map = {1: "", 2: "", 3: "", 4: "", 5: ""}
            full_label = f"{step_label}"
            
            # Primary type for active, Secondary for inactive
            if st.button(full_label, key=f"flow_btn_{step_id}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state['ml_step'] = step_id
                st.rerun()
        
        # Draw arrow in next column if not last step
        if idx < len(steps) - 1:
            with cols[col_idx + 1]:
                st.markdown("<h3 style='text-align: center; margin: 0; color: #b0b2b6;'>➜</h3>", unsafe_allow_html=True)

    st.divider()

    # Content Rendering
    current = st.session_state['ml_step']

    if current == 1:
        st.header("1. Collecte & Préparation Initiale")
        st.info("Stratégie : Créer une base solide et représentative pour l'entraînement.")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
            **Données & Split :**
            - Provenance : Datasets publics (PlanteNet).
            - **Split Stratifié : 80 / 10 / 10**
                - **Train** : Pour l'entraînement.
                - **Validation** : Pour le tuning des hyperparamètres.
                - **Test** : Pour l'évaluation finale objective.
            """)
            
        with c2:
            st.markdown("""
            **Objectifs :**
            - Images nettes et représentatives.
            - Equilibrage des classes via rééchantillonnage (sur Train uniquement).
            """)


        st.divider()
        st.write("### Exploration par l'équipe")
        cols_team = st.columns(4)
        team_data = [
            ("SVM (RBF)", "Bernadette GASMI"),
            ("XGBoost", "Lionel SCHNEIDER"),
            ("Reg. Logistique", "JB MACK"),
            ("Extra-Trees", "Morgan PERCHEC")
        ]
        for idx, (algo, author) in enumerate(team_data):
            with cols_team[idx]:
                st.info(f"**{algo}**\n\n*{author}*")
            
    elif current == 2:
        st.header("2. Extraction de Features (Handcrafted)")
        c_text, c_img = st.columns([1, 2])
        with c_text:
            st.markdown("""
            Calcul de **34 descripteurs numériques** via **OpenCV**, **NumPy** et **skimage.feature**.
            Cette étape transforme les pixels bruts en vecteurs d'information exploitables par des modèles classiques.
            """)
            st.markdown("""
            - **Méthodes utilisées :**
                - **Morphologie** : Forme de la feuille (Aire, périmètre, circularité).
                - **Couleur** : Statistiques RGB/HSV (Moyennes, Écarts-types).
                - **Texture** : Matrices de co-occurrence (Haralick/GLCM), contraste, uniformité.
                - **Fréquences** : Spectre de Fourier (FFT).
            """)
        
        with c_img:
            st.image("Streamlit/assets/Les datasets/Caractéristiques.drawio.png", caption="Synthèse des Features", width=700)
            
        ranking_path = "results/feature_ranking.csv"
        if os.path.exists(ranking_path):
            st.divider()
            st.markdown("### Importance des Features")
            df_rank = pd.read_csv(ranking_path).head(15).sort_values(by="final_score", ascending=True)
            fig_rank = px.bar(df_rank, x="final_score", y="feature", orientation="h",
                               title="Top 15 des Features les plus discriminantes",
                               color="final_score", color_continuous_scale="GnBu")
            st.plotly_chart(fig_rank, use_container_width=True)

    elif current == 3:
        st.header("3. Nettoyage & Augmentation")
        
        col_aug1, col_aug2 = st.columns(2)
        with col_aug1:
            st.subheader("Pré-traitements")
            st.write("Suppression des images corrompues ou générant des valeurs NaN après extraction.")
            
        with col_aug2:
            st.subheader("Data Augmentation")
            st.markdown("""
            **Enrichissement du dataset (sur Train uniquement) :**
            - Rotations, Flips, Bruit gaussien.
            - **Volume final** : 91 770 images.
            """)
            st.info("Permet de renforcer la robustesse du modèle face aux variations.")

    elif current == 4:
        st.header("4. Scaling & Sélection")
        
        st.markdown("### Standardisation")
        st.write("Utilisation de **RobustScaler** pour normaliser les caractéristiques.")
        st.success("**Pourquoi ?** Pour gérer efficacement les **40% d'outliers** détectés dans les distributions de features.")
        
        st.divider()
        
        st.markdown("### Sélection de Features (SHAP)")
        st.write("Identification des features les plus informatives pour réduire la dimensionnalité sans perdre de précision.")
        
        shap_global_path = "figures/shap_analysis/1_global_importance.png"
        if os.path.exists(shap_global_path):
             st.image(shap_global_path, caption="Importance Globale des Features (SHAP)", use_container_width=True)

    elif current == 5:
        st.header("5. Modélisation & Évaluation")
        
        st.markdown("""
        Entraînement sur le **Train Set**, validation sur le **Valid Set**, et verdict final sur le **Test Set**.
        """)
        
        with st.expander("Définitions des Métriques", expanded=False):
            st.markdown("""
            - **Accuracy** : % de prédictions correctes.
            - **Précision** : Capacité à éviter les faux positifs (fiabilité).
            - **Rappel** : Capacité à détecter tous les cas réels (exhaustivité).
            - **F1-score** : Moyenne harmonique (équilibre) entre précision et rappel.
            """)
        


        st.divider()
        st.success("**Résultats sur le Test Set**")
        
        # Reuse logic for reports logic...

        # Données complètes (Source: machine_learning.py)
        full_perf_data = {
            "Modèle": ["SVM (RBF)", "XGBoost", "Reg. Logistique", "Extra-Trees"],
            "Accuracy": [0.9370, 0.9038, 0.8615, 0.8310],
            "Précision (macro)": [0.9271, 0.9051, 0.8462, 0.8607],
            "Rappel (macro)": [0.9207, 0.8654, 0.8214, 0.7405],
            "F1-score (macro)": [0.9237, 0.8839, 0.8328, 0.7863]
        }
        df_full = pd.DataFrame(full_perf_data)
        
        st.write("### Performances Globales")
        df_melt = df_full.melt(id_vars="Modèle", var_name="Métrique", value_name="Valeur")
        fig_full = px.bar(df_melt, x="Modèle", y="Valeur", color="Métrique", barmode="group",
                        text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_full.update_layout(yaxis_range=[0.7, 1.0], margin=dict(t=0, b=0, l=0, r=0), 
                               legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_full, use_container_width=True)
        
        st.divider()
        st.subheader("Détail par classe & Matrices de Confusion")
        
        model_paths = {
            "SVM (RBF)": {
                "report": "results/Machine_Learning/svm_rbf_baseline_features_selected/evaluation/baseline/classification_report.txt",
                "plot": "results/Machine_Learning/svm_rbf_baseline_features_selected/plots/baseline/confusion_matrix.png"
            },
            "XGBoost": {
                "report": "results/Machine_Learning/xgb_baseline/evaluation/xgboost/classification_report.txt",
                "plot": "results/Machine_Learning/xgb_baseline/plots/xgboost/confusion_matrix.png"
            },
            "Reg. Logistique": {
                "report": "results/Machine_Learning/logreg_baseline/evaluation/logreg/classification_report.txt",
                "plot": "results/Machine_Learning/logreg_baseline/plots/logreg/confusion_matrix.png"
            },
            "Extra-Trees": {
                "report": "results/Machine_Learning/extra_trees_baseline/evaluation/extra_trees/classification_report.txt",
                "plot": "results/Machine_Learning/extra_trees_baseline/plots/extra_trees/confusion_matrix.png"
            }
        }

        cols_reports = st.columns(2)
        for idx, (model_name, paths) in enumerate(model_paths.items()):
            # Distribute in columns
            with cols_reports[idx % 2]:
                with st.expander(f"{model_name}", expanded=False):
                    
                    # 1. Classification Report
                    if os.path.exists(paths["report"]):
                        df_report = parse_classification_report(paths["report"])
                        if df_report is not None:
                            df_display = df_report.copy()
                            for col in ["Précision", "Rappel", "F1-score"]:
                                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")
                            
                            st.caption("Rapport de Classification")
                            img_buf = render_df_as_image(df_display, title=None)
                            if img_buf:
                                st.image(img_buf, use_container_width=True)
                    
                    # 2. Confusion Matrix
                    if os.path.exists(paths["plot"]):
                        st.caption("Matrice de Confusion")
                        st.image(paths["plot"], use_container_width=True)
                    else:
                        st.warning("Matrice de confusion introuvable")

def sidebar_choice():
    st.title("Machine Learning")
    render_roadmap()
