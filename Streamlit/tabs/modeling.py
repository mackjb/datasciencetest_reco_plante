import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# =========================
# CONTENU MACHINE LEARNING
# =========================
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

    tabs = st.tabs(["⚙️ Features", "📊 Performances", "🧠 SHAP"])
    
    with tabs[0]:
        st.header("1. Extraction des Descripteurs")
        st.markdown("""
        **Catégories extraites :**
        - 📏 **Morphologie** : Aire, périmètre, circularité, excentricité, aspect ratio, densité de contours
        - 🎨 **Couleur** : Moyennes et Écarts-types RGB / HSV
        - 🕸️ **Texture** : Matrices de co-occurrence (GLCM) - netteté, contraste, energy, homogeneity, dissimilarity, correlation
        - 🔄 **Invariants** : Moments de Hu (hu_1 à hu_7)
        - 📻 **Fréquences** : Entropie et puissance spectrale (FFT)
        - 📐 **Gradients** : Descripteurs HOG (moyenne, écart-type, entropie)
        
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
            st.plotly_chart(fig_rank, use_container_width=True)
        
    with tabs[1]:
        st.header("2. Analyse des Performances")
        
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
        
        st.markdown("""
        **Observations clés :**
        - La **contour_density** domine très nettement l'importance globale (30% supérieure à la 2ème feature)
        - Les features de **fréquence spectrale** (fft_entropy) et de **couleur** (mean_R, mean_B) complètent le trio de tête
        - Chaque classe de maladie s'appuie sur un **sous-ensemble différent de features**
        - Les **34 features extraites sont toutes pertinentes**, aucune n'est totalement négligeable
        """)
        
        shap_dir = "figures/shap_analysis"
        p1 = os.path.join(shap_dir, "1_global_importance.png")
        if os.path.exists(p1):
            st.image(p1, caption="Importance Globale des Features (Top 25)", use_container_width=True)
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

# =========================
# CONTENU DEEP LEARNING
# =========================
def render_dl_content():
    st.markdown("""
    Le Deep Learning permet d'apprendre automatiquement les features directement à partir des pixels, 
    contrairement au Machine Learning classique qui nécessite une extraction manuelle de descripteurs.
    """)
    
    # --- Méthodologie ---
    with st.expander("🎯 Méthodologie & Critères de Sélection", expanded=True):
        st.markdown("""
        ### Démarche structurée en 3 étapes :
        
        1. **Exploration** : 9 architectures testées pour comprendre le Deep Learning et ses défis
        2. **Évaluation comparative** : Restriction à quelques architectures couvrant 3 cas d'usage
        3. **Sélection & Recommandation** : Projection pour un déploiement réel
        
        ### Critères de sélection :
        
        | Catégorie | Critères | Justification |
        |-----------|----------|---------------|
        | **Métier** | Précision (Macro-F1, Accuracy) | Capacité à bien prédire toutes les classes |
        | | Généralisation (écart val/test) | Robustesse du modèle (<2% = bon, >5% = overfitting) |
        | | Couverture opérationnelle | Réponse aux 3 cas d'usage métier |
        | **Technique** | Coût d'inférence (FLOPs, latence) | Impact sur batterie et expérience utilisateur |
        | | Coût d'entraînement (temps, GPU) | Budget cloud et itérations rapides |
        | | Complexité (paramètres, maintenabilité) | Taille du modèle et facilité de maintenance |
        | **Autres** | Interprétabilité | Capacité à expliquer les prédictions (Grad-CAM) |
        | | Besoins en données | Quantité d'images annotées nécessaire |
        """)
        
        st.info("""
        **🎯 Les 3 cas d'usage :**
        - **Cas 1** : Identification d'espèce uniquement
        - **Cas 2** : Diagnostic ciblé (espèce connue → maladie)
        - **Cas 3** : Diagnostic complet (espèce + maladie inconnues)
        """)

    # Onglets principaux DL
    dl_tabs = st.tabs(["🏗️ Architectures", "📊 Performances", "🧬 Démos Interactives"])
    
    with dl_tabs[0]:
        st.header("Exploration des 9 Architectures")
        
        st.markdown("""
        **Protocole expérimental commun :**
        - Dataset : PlantVillage/color
        - Backbone pré-entraîné : **EfficientNetV2S** (ImageNet)
        - Splits identiques pour tous les modèles
        - Hyperparamètres fixés : learning rate, batch size, augmentation
        - Métriques : Loss, Accuracy, Macro-F1, matrice de confusion
        """)
        
        st.divider()
        
        # Présentation des architectures
        arch_info = [
            {
                "num": "1",
                "nom": "Trois modèles indépendants",
                "desc": "3 CNN spécialisés (species, health, disease)",
                "avantages": "Simplicité, performances maximales par tâche",
                "limites": "Triplication des ressources, pas de synergie"
            },
            {
                "num": "2",
                "nom": "Deux modèles (species + disease_extended)",
                "desc": "'Healthy' intégré comme maladie spéciale",
                "avantages": "Diagnostic complet en 2 inférences",
                "limites": "Déséquilibre accru, perte de métrique binaire"
            },
            {
                "num": "3",
                "nom": "Modèle unifié (35 classes)",
                "desc": "Étiquette combinée Espèce__État",
                "avantages": "Un seul modèle, synergie entre tâches",
                "limites": "Moins flexible, classes rares sous-apprises"
            },
            {
                "num": "4",
                "nom": "Architecture en cascade",
                "desc": "Species → Disease avec attention spatiale",
                "avantages": "Prédiction guidée, attention spatiale",
                "limites": "Propagation d'erreurs, latence accrue"
            },
            {
                "num": "5",
                "nom": "CNN + SVM",
                "desc": "Embeddings CNN + classifieurs SVM",
                "avantages": "Entraînement rapide, simplicité",
                "limites": "Features génériques, pas d'adaptation"
            },
            {
                "num": "6",
                "nom": "Multi-tâche unifié (3 têtes)",
                "desc": "Backbone partagé + 3 têtes parallèles",
                "avantages": "Synergie, une seule inférence",
                "limites": "Conflits d'optimisation, pas de fine-tuning"
            },
            {
                "num": "7",
                "nom": "Multi-tâche 2 têtes + signal santé",
                "desc": "Species + Disease avec signal santé auxiliaire",
                "avantages": "Synergie, masquage des 'healthy'",
                "limites": "Pas de sortie santé explicite"
            },
            {
                "num": "8",
                "nom": "Multi-tâche simplifié",
                "desc": "Species + Disease (incluant healthy)",
                "avantages": "Simplicité, cohérence de décision",
                "limites": "Déséquilibre 'healthy', pas de conditionnement"
            },
            {
                "num": "9",
                "nom": "Architecture conditionnée",
                "desc": "Disease conditionnée par Species + Health",
                "avantages": "Conditionnement explicite, synergie",
                "limites": "Propagation d'erreurs, pas de sortie santé"
            }
        ]
        
        for arch in arch_info:
            with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                st.markdown(f"**Description** : {arch['desc']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"✅ **Avantages** : {arch['avantages']}")
                with col2:
                    st.markdown(f"⚠️ **Limites** : {arch['limites']}")
    
    with dl_tabs[1]:
        st.header("Synthèse des Performances")
        
        # Tableau de performances
        perf_dl = {
            "Architecture": ["Archi 1", "Archi 2", "Archi 3", "Archi 4", "Archi 5", "Archi 6", "Archi 7", "Archi 8", "Archi 9"],
            "Species Macro-F1": [0.990, 0.990, 0.990, 0.990, 0.985, 0.988, 0.990, 0.989, 0.990],
            "Species Accuracy": [0.990, 0.990, 0.990, 0.990, 0.986, 0.988, 0.990, 0.989, 0.990],
            "Disease Accuracy": [0.990, 0.988, 0.990, 0.987, 0.982, 0.975, 0.990, 0.986, 0.990],
            "FLOPs (relatif)": ["3×", "2×", "1×", "2×", "1×", "1×", "1×", "1×", "1×"],
            "Maintenabilité": ["Faible", "Moyenne", "Élevée", "Faible", "Moyenne", "Moyenne", "Moyenne", "Moyenne", "Faible"]
        }
        df_perf_dl = pd.DataFrame(perf_dl)
        
        st.dataframe(df_perf_dl.style.highlight_max(subset=["Species Macro-F1", "Species Accuracy", "Disease Accuracy"], axis=0))
        
        st.divider()
        
        st.markdown("""
        ### 🎯 Décisions et Exclusions
        
        **Architectures exclues :**
        - **Archi 4** : Cascade complexe sans gain tangible, risque de propagation d'erreurs
        - **Archi 6** : En retrait sur la maladie (0.975 vs ≥0.989 pour les autres)
        - **Archi 8** : Pas de bénéfice mesurable vs Archi 7/9
        
        **Architectures retenues pour recommandation :**
        - **Archi 3** : Excellente simplicité de déploiement (1 modèle, 1 inférence)
        - **Archi 7** : Bon compromis performance/efficience
        - **Archi 9** : Conditionnement explicite, synergie maximale
        """)
        
        # Graphique comparatif
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Species Macro-F1',
            x=df_perf_dl['Architecture'],
            y=df_perf_dl['Species Macro-F1'],
            marker_color='lightblue'
        ))
        fig_comp.add_trace(go.Bar(
            name='Disease Accuracy',
            x=df_perf_dl['Architecture'],
            y=df_perf_dl['Disease Accuracy'],
            marker_color='lightcoral'
        ))
        fig_comp.update_layout(
            title="Comparaison des Performances par Architecture",
            yaxis_range=[0.97, 1.0],
            barmode='group'
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with dl_tabs[2]:
        st.header("🧬 Démos Interactives")
        
        st.info("""
        💡 **Les démos interactives complètes sont disponibles dans l'onglet "🧠 Deep Learning" de la sidebar.**
        
        Vous y trouverez :
        - **Archi 3** : 6 images de plantes malades avec diagnostic de maladies
        - **Archi 9** : 6 images de plantes saines avec identification d'espèces
        - Visualisation Grad-CAM pour l'interprétabilité
        """)
        
        st.markdown("""
        ### Aperçu des Architectures Retenues
        
        #### 🏆 Architecture 3 : Solution Edge/Mobile
        - **Type** : Modèle unifié (35 classes combinées Espèce__État)
        - **Avantages** : Un seul modèle, une seule inférence, déploiement simplifié
        - **Performance** : Species Macro-F1 = 0.990, Disease Accuracy = 0.990
        - **Cas d'usage** : Idéal pour applications mobiles (smartphones, tablettes)
        
        #### 🚀 Architecture 9 : Solution Production Cloud
        - **Type** : Architecture conditionnée (Disease guidée par Species + Health)
        - **Avantages** : Conditionnement explicite, synergie maximale entre tâches
        - **Performance** : Species Macro-F1 = 0.990, Disease Accuracy = 0.990
        - **Cas d'usage** : Déploiement cloud avec ressources importantes
        """)

# =========================
# FONCTION PRINCIPALE
# =========================
def sidebar_choice():
    st.title("📊 Modélisation")
    
    # Création des deux sous-onglets
    main_tabs = st.tabs(["🤖 Machine Learning", "🧠 Deep Learning"])
    
    with main_tabs[0]:
        render_ml_content()
    
    with main_tabs[1]:
        render_dl_content()
