import streamlit as st
import pandas as pd
import plotly.express as px
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


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
            3. 🧹 **Nettoyage** : Suppression des images corrompues (9 images avec NaN).
            4. 📈 **Augmentation** : Enrichissement du dataset → **91 770 images finales**.
            5. ⚖️ **Scaling** : Normalisation **RobustScaler** (gestion des 40% d'outliers).
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

    tabs = st.tabs(["Features", "Performances", "SHAP"])
    
    with tabs[0]:
        st.header("Extraction des Descripteurs")

        col_p1, col_p2, = st.columns(2)
        with col_p1:
            st.subheader(" ")       
            st.image(
                os.path.join(ASSETS_DIR, "Les datasets/Caractéristiques.drawio.png"),
                caption="Synthèse des catégories de descripteurs extraits",
                width=700,
            )

        with col_p2:
            st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
            st.markdown("""
            **Catégories extraites :**
            - **Morphologie** : Aire, périmètre, circularité, excentricité, aspect ratio, densité de contours
            - **Couleur** : Moyennes et Écarts-types RGB / HSV
            - **Texture** : Matrices de co-occurrence (GLCM) - netteté, contraste, energy, homogeneity, dissimilarity, correlation
            - **Invariants** : Moments de Hu (hu_1 à hu_7)
            - **Fréquences** : Entropie et puissance spectrale (FFT)
            - **Gradients** : Descripteurs HOG (moyenne, écart-type, entropie)
            """)

            st.markdown("""
            Ces descripteurs sont concaténés pour former un **vecteur unique par image** (34 features), 
            servant ensuite d'entrée aux algorithmes de classification.
            """)

        st.divider()
        st.subheader("Importance des Features")
        ranking_path = "results/feature_ranking.csv"
        if os.path.exists(ranking_path):
            df_rank = pd.read_csv(ranking_path).head(15).sort_values(by="final_score", ascending=True)
            fig_rank = px.bar(df_rank, x="final_score", y="feature", orientation="h",
                               title="Top 15 des Features les plus discriminantes",
                               color="final_score", color_continuous_scale="GnBu")
            st.plotly_chart(fig_rank, width=800)
        
    with tabs[1]:
        st.header("Analyse des Performances")
        
        st.markdown("""
        Résultats obtenus pour l'**Objectif 1** (Identification de l'espèce) sur l'ensemble de test.
        Nous avons comparé **4 modèles principaux**.
        """)
        
        perf_data = {
            "Modèle": ["SVM (RBF)", "XGBoost", "Reg. Logistique", "Extra-Trees"],
            "Accuracy": [0.9370, 0.9038, 0.8615, 0.8310],
            "F1-score (macro)": [0.9237, 0.8839, 0.8328, 0.7863]
        }
        df_perf = pd.DataFrame(perf_data)
        
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.dataframe(df_perf.style.apply(lambda x: ['background-color: yellow' if x.name == 0 else '' for _ in x], axis=1))
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
        st.header("Interprétabilité SHAP")
        col_shap_1, col_shap_2 = st.columns(2)

        with col_shap_1:
        
            shap_dir = "figures/shap_analysis"
            p1 = os.path.join(shap_dir, "1_global_importance.png")
            if os.path.exists(p1):
                st.image(p1,  width=800)
            else:
                st.warning("Graphique SHAP non trouvé.")
            
            st.markdown("""



            **Observations clés :**
            - La **contour_density** domine très nettement l'importance globale (30% supérieure à la 2ème feature)
            - Les features de **fréquence spectrale** (fft_entropy) et de **couleur** (mean_R, mean_B) complètent le trio de tête
            - Chaque classe de maladie s'appuie sur un **sous-ensemble différent de features**
            - Les **34 features extraites sont toutes pertinentes**, aucune n'est totalement négligeable
            """)

        with col_shap_2:

            shap_dir = "figures/shap_analysis"
            p2 = os.path.join(shap_dir, "3_top_features_by_class.png")
            if os.path.exists(p2):
                st.image(p2,width=700)
            else:
                st.warning("Graphique SHAP non trouvé.")
        
            st.markdown("""
            **Observations clés :**
            - L'analyse par classe révèle des signatures de features distinctes pour chaque maladie : 
            par exemple, hog_std (texture) est extrêmement discriminant pour Apple_scab mais beaucoup moins pour les autres maladies. 
            À l'inverse, contour_density présente une importance élevée et relativement uniforme pour plusieurs maladies, indiquant 
            qu'il s'agit d'une feature généraliste importante pour détecter les anomalies foliaires. 
            Cette variabilité confirme que différentes maladies se manifestent par des combinaisons spécifiques de caractéristiques visuelles.
            """)
            
        st.divider()
        st.subheader("Synthèse des Résultats par Modèle")
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
        
        st.info("**Constat** : Le **SVM (RBF)** surpasse ses concurrents sur toutes les métriques, confirmant sa robustesse face au déséquilibre des classes.")


def sidebar_choice():
    st.title("Machine Learning")
    render_ml_content()

