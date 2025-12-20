import streamlit as st
import pandas as pd
import plotly.express as px
import os

def sidebar_choice():
    st.title("🤖 Modélisation Machine Learning Classique")
    
    st.markdown("""
    Avant d'utiliser des réseaux de neurones profonds, nous avons établi une **baseline** avec des algorithmes de Machine Learning classiques.
    L'approche repose sur l'extraction manuelle de features (Handcrafted Features).
    
    ### 🎯 Les 3 Objectifs Visés
    1.  **Objectif 1** : Identification de la Plante (Espèce).
    2.  **Objectif 2** : Détection de l'État de Santé (Sain vs Malade).
    3.  **Objectif 3** : Diagnostic Spécifique de la Maladie.
    """)
    
    tab1, tab2, tab3 = st.tabs(["⚙️ Feature Engineering", "📊 Performance Modèles", "🧠 Interprétabilité (SHAP)"])
    
    with tab1:
        st.header("Extraction de Caractéristiques")
        st.markdown("""
        Nous transformons chaque image en un vecteur de données structurées pour nourrir nos classifieurs.
        
        | Type | Descripteurs | Dimension |
        | :--- | :--- | :--- |
        | **Forme** | Moments de Hu, Aire, Périmètre | Faible |
        | **Texture** | Haralick (GLCM), LBP | Moyenne |
        | **Couleur** | Histogrammes RGB/HSV, Momente | Faible |
        | **Fréquentiel** | HOG (Histogram of Oriented Gradients) | Élevée |
        """)
        
        st.divider()
        st.subheader("Classement des Caractéristiques (Feature Ranking)")
        st.markdown("Voici l'importance relative des descripteurs extraits pour la classification.")
        
        ranking_path = "results/feature_ranking.csv"
        if os.path.exists(ranking_path):
            df_rank = pd.read_csv(ranking_path)
            # On ne garde que les 20 premières si il y en a trop
            df_plot = df_rank.head(20).sort_values(by="final_score", ascending=True)
            
            fig_rank = px.bar(df_plot, 
                              x="final_score", 
                              y="feature", 
                              orientation="h",
                              title="Top 20 des Features (Score Final)",
                              labels={"final_score": "Importance (0-1)", "feature": "Caractéristique"},
                              color="final_score",
                              color_continuous_scale="Viridis")
            
            fig_rank.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_rank, use_container_width=True)
            
            st.info("💡 **Observations** : La **luminosité (mean_B)** et la **dissimilarité** de texture sortent souvent en tête, confirmant l'impact de l'éclairage et de la régularité du limbe.")
        else:
            st.warning("Fichier feature_ranking.csv non trouvé.")
        
    with tab2:
        st.header("Analyse de Performance")
        
        st.markdown("""
        Voici les résultats obtenus pour l'**Objectif 1** (Identification de l'espèce) sur l'ensemble de test.
        Nous avons comparé 4 modèles principaux.
        """)
        
        # Tableau des performances du rapport
        perf_data = {
            "Modèle": ["SVM (RBF)", "XGBoost", "Régression Logistique", "Extra-Trees"],
            "Accuracy": [0.9370, 0.9038, 0.8615, 0.8310],
            "F1-score (macro)": [0.9237, 0.8839, 0.8328, 0.7863]
        }
        df_perf = pd.DataFrame(perf_data)
        st.table(df_perf)

        st.info("💡 **Constat** : Le **SVM (RBF)** se détache nettement par sa capacité à capturer les relations non-linéaires entre les descripteurs morphologiques et colorimétriques.")

        # Affichage conditionnel des résultats
        res_dir = "results/Machine_Learning/logreg_baseline/plots/logreg/"
        if os.path.exists(res_dir):
            cm_path = os.path.join(res_dir, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Matrice de Confusion (Baseline)", use_container_width=True)
                
    with tab3:
        st.header("Importance des Features (SHAP)")
        st.write("Analyse de l'impact des descripteurs sur la décision du modèle.")
        
        col_shap1, col_shap2 = st.columns(2)
        
        shap_dir = "figures/shap_analysis"
        
        with col_shap1:
            p1 = os.path.join(shap_dir, "1_global_importance.png")
            if os.path.exists(p1):
                st.image(p1, caption="Importance Globale des Features", use_container_width=True)
            else:
                st.write("Graphique Global manquant")
                
        with col_shap2:
            p2 = os.path.join(shap_dir, "2_feature_impact_summary.png")
            if os.path.exists(p2):
                st.image(p2, caption="Impact détaillé (Beeswarm Plot)", use_container_width=True)
            else:
                st.write("Graphique Impact manquant")

