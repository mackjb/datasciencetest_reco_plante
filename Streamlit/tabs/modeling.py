import streamlit as st
import pandas as pd
import plotly.express as px
import os

def render_ml_content():
    st.markdown("""
    L'approche classique repose sur l'**extraction manuelle de descripteurs** (Handcrafted Features) plutôt que sur l'apprentissage direct des pixels. 
    Elle sert de **baseline** robuste pour comparer nos futurs modèles Deep Learning.
    """)
    
    # --- Méthodologie ---
    with st.expander("🛠️ Méthodologie & Pipeline", expanded=True):
        col_m1, col_m2 = st.columns([1.5, 1])
        
        with col_m1:
            st.markdown("""
            **Étapes clefs du pipeline robuste :**
            1. 📸 **Collecte** : Images nettes et segmentées.
            2. 📐 **Extraction** : Calcul des descripteurs (Morpho, Couleur, Texture...).
            3. 🧹 **Nettoyage** : Suppression des images corrompues.
            4. 📈 **Augmentation** : Enrichissement du dataset (Rotation, Flip...).
            5. ⚖️ **Scaling** : Normalisation **RobustScaler** (gestion des outliers).
            6. 🎯 **Sélection** : Garder les features les plus discriminantes (SHAP).
            7. 🤖 **Modélisation** : Entraînement des classifieurs.
            """)
            
        with col_m2:
            st.info("""
            **🎯 Exploration par l'équipe :**
            - **SVM (RBF)** : Bernadette GASMI
            - **XGBoost** : Lionel SCHNEIDER
            - **Reg. Logistique** : JB MACK
            - **Extra-Trees** : Morgan PERCHEC
            """)
        
        st.divider()
        st.info("""
        **Points clés** : 
        - Split **80/10/10** stratifié.
        - **Data Augmentation** sur le train (91 770 images finales).
        - **RobustScaler** utilisé pour gérer les 40% d'outliers détectés.
        """)

    tabs = st.tabs(["⚙️ Features", "📊 Performances", "🧠 SHAP"])
    
    with tabs[0]:
        st.header("1. Extraction des Descripteurs")
        st.markdown("""
        **Catégories extraites :**
        - 📏 **Morphologie** : Aire, périmètre, circularité...
        - 🎨 **Couleur** : Moyennes et Écarts-types RGB / HSV.
        - 🕸️ **Texture** : Matrices de co-occurrence (GLCM).
        - 🔄 **Invariants** : Moments de Hu.
        - 📻 **Fréquences** : Entropie et puissance spectrale (FFT).
        - 📐 **Gradients** : Descripteurs HOG.
        """)

        st.divider()
        st.subheader("Importance des Features")
        ranking_path = "results/feature_ranking.csv"
        if os.path.exists(ranking_path):
            df_rank = pd.read_csv(ranking_path).head(15).sort_values(by="final_score", ascending=True)
            fig_rank = px.bar(df_rank, x="final_score", y="feature", orientation="h",
                               title="Top 15 des Features les plus discriminantes",
                               color="final_score", color_continuous_scale="GnBu")
            st.plotly_chart(fig_rank, use_container_width=True)
        
    with tabs[1]:
        st.header("2. Analyse des Performances")
        perf_data = {
            "Modèle": ["SVM (RBF)", "XGBoost", "Reg. Logistique", "Extra-Trees"],
            "Accuracy": [0.9370, 0.9038, 0.8615, 0.8310],
            "F1-score (macro)": [0.9237, 0.8839, 0.8328, 0.7863]
        }
        df_perf = pd.DataFrame(perf_data)
        
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.dataframe(df_perf.style.highlight_max(axis=0))
            st.success("🏆 **SVM (RBF)** est le plus performant.")
        
        with col2:
            fig_perf = px.bar(df_perf, x="Modèle", y="F1-score (macro)", color="Modèle",
                               title="Comparaison des F1-Scores", text_auto='.2f')
            fig_perf.update_layout(showlegend=False)
            st.plotly_chart(fig_perf, use_container_width=True)

        cm_path = "results/Machine_Learning/svm_rbf_baseline_features_selected/plots/baseline/confusion_matrix.png"
        if os.path.exists(cm_path):
            with st.expander("🔍 Voir la Matrice de Confusion (SVM-RBF)"):
                st.image(cm_path, use_container_width=True)
                
    with tabs[2]:
        st.header("3. Interprétabilité SHAP")
        shap_dir = "figures/shap_analysis"
        p1 = os.path.join(shap_dir, "1_global_importance.png")
        if os.path.exists(p1):
            st.image(p1, caption="Importance Globale des Features (Top 25)", use_container_width=True)
            st.write("**Observation clé** : La `contour_density` est le descripteur le plus influent.")
        else:
            st.warning("Graphique SHAP non trouvé.")

        st.divider()
        st.subheader("🏆 Synthèse des Résultats par Modèle")
        st.markdown("Comparaison finale des performances sur l'Objectif 1 (Identification de l'espèce).")
        
        full_perf_data = {
            "Modèle": ["SVM (RBF)", "XGBoost", "Reg. Logistique", "Extra-Trees"],
            "Accuracy": [0.9370, 0.9038, 0.8615, 0.8310],
            "Précision (macro)": [0.9271, 0.9051, 0.8462, 0.8607],
            "Rappel (macro)": [0.9207, 0.8654, 0.8214, 0.7405],
            "F1-score (macro)": [0.9237, 0.8839, 0.8328, 0.7863]
        }
        df_full = pd.DataFrame(full_perf_data)
        df_melt = df_full.melt(id_vars="Modèle", var_name="Métrique", value_name="Valeur")
        
        fig_full = px.bar(df_melt, x="Modèle", y="Valeur", color="Métrique", barmode="group",
                          title="Comparaison Multi-Métriques (Test Set)",
                          text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Pastel)
        
        fig_full.update_layout(yaxis_range=[0.7, 1.0])
        st.plotly_chart(fig_full, use_container_width=True)
        
        st.info("💡 **Constat** : Le **SVM (RBF)** surpasse ses concurrents sur toutes les métriques, confirmant sa robustesse face au déséquilibre des classes.")

def sidebar_choice():
    st.title("📊 Modélisation")
    render_ml_content()
