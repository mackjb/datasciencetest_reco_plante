import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import io
import base64




def parse_classification_report(file_path):
    """Parses a sklearn classification report text file into a DataFrame."""
    try:
        with open(file_path, "r") as f:
            report_text = f.read()
        
        # Split lines and filter empty ones
        lines = [line.strip() for line in report_text.split('\n') if line.strip()]
        
        # Parse data
        data = []
        for line in lines[1:]: # Skip header
             parts = line.split()
             if len(parts) >= 2: # Check for valid line
                # Handle class names with spaces or special chars if any, though report usually aligns well
                # The last 3 are metrics, 4th from last is support, rest is name
                if parts[0] in ['accuracy', 'macro', 'weighted']: # Skip summary rows for species table
                     continue
                
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

import matplotlib
matplotlib.use("Agg") # Force headless backend
import matplotlib.pyplot as plt
import seaborn as sns

def render_df_as_image(df, title=None):
    """Renders a DataFrame as a matplotlib figure image."""
    try:
        # Create figure using Figure constructor to avoid pyplot global state issues if possible, 
        # but subplots is often easier. Let's stick to subplots but use fig methods.
        fig, ax = plt.subplots(figsize=(5, len(df) * 0.25 + 1)) # Adjust height based on rows
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
        
        # Style
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5) # Scale width, height
        
        if title:
            fig.suptitle(title, fontweight="bold")
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig) # Explicitly close via pyplot to free state
        buf.seek(0)
        return buf
    except Exception as e:
        return f"Error: {str(e)}"






def render_ml_content():
    st.markdown("""
    Les modèles ML ne traitent pas directement les pixels bruts d’une image. Il faut **extraire des caractéristiques numériques**, 
    pour les utiliser comme entrées et effectuer une exploration statistique.  
    Cette approche permet de comprendre les attributs visuels déterminants (forme, texture, couleur) avant d'aborder des modèles plus complexes.
    """)
    
    # --- Méthodologie ---
    with st.expander("Méthodologie & Pipeline", expanded=True):
        col_m1, col_m2 = st.columns([1.5, 1])
        
        with col_m1:
            st.markdown("""
            **Étapes clefs du pipeline robuste :**
            1. **Extraction** : Calcul des 34 descripteurs numériques (Morpho, Couleur, Texture...) via **OpenCV**, **NumPy** et **skimage.feature**.
            2. **Split** : Division en **Train / Valid / Test** pour une évaluation rigoureuse.
            3. **Rééchantillonnage** : Équilibrage des classes (sur *Train* uniquement).
            4. **Pré-traitements** : 
                *   **Augmentation** (sur *Train*) : Enrichissement du dataset.
                *   **Scaling** : Standardisation / Normalisation des caractéristiques.
            5. **Sélection** : Identification des features informatives avec **SHAP**.
            6. **Modélisation** : Entraînement sur *Train*, validation sur *Valid*.
            7. **Évaluation** : Mesure de la performance sur *Test* (Précision, Rappel, F1-score).
                *   **Accuracy** : % de prédictions correctes.
                *   **Précision** : Capacité à éviter les faux positifs (fiabilité de la détection).
                *   **Rappel** : Capacité à détecter tous les cas réels (exhaustivité).
                *   **F1-score** : Équilibre entre précision et rappel (score global).
            """)
            
        with col_m2:
            st.info("""
            **Exploration par l'équipe :**
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
        st.header("1. Extraction des Descripteurs")

        st.markdown("""
        <style>
        [data-testid="stImage"] img {
            transition: transform 0.4s ease;
            z-index: 1;
        }
        [data-testid="stImage"] img:hover {
            transform: scale(1.7); /* Revert to standard zoom for Features tab */
            z-index: 1000;
            cursor: zoom-in;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        
        /* Custom class for performance images */
        .perf-zoom {
            transition: transform 0.4s ease;
            border-radius: 5px;
            cursor: zoom-in;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .perf-zoom:hover {
            transform: scale(4.5); /* Stronger zoom for detailed tables */
            z-index: 9999;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            position: relative; 
        }
        </style>
        """, unsafe_allow_html=True)
        
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            st.image(
                "Streamlit/assets/Les datasets/Caractéristiques.drawio.png",
                caption="Synthèse des catégories de descripteurs extraits",
                width=750,
            )

        st.markdown("""
        **Catégories extraites :**
        - **Caractéristiques morphologiques** : Superficie (aire), périmètre, circularité, excentricité, rapport d’aspect (aspect ratio) et densité de contours. Ces indicateurs décrivent la **forme globale** des objets présents sur l’image.
        - **Caractéristiques colorimétriques** : Moyennes et écarts-types des canaux RGB (mean/std_R, G, B), ainsi que les moyennes des composantes HSV (mean_H, S, V). Elles permettent de représenter les **couleurs dominantes** et leur variation.
        - **Caractéristiques de texture** : Netteté, contraste, energy, homogeneity, dissimilarity, correlation. Calculées à partir de matrices de co-occurrence (GLCM), elles décrivent les **variations locales d’intensité**.
        - **Descripteurs invariants** : Les moments de Hu (hu_1 à hu_7) capturent la forme de manière **invariante** à la rotation, à la translation et au changement d’échelle.
        - **Descripteurs fréquentiels** : Coefficients de la transformée de Fourier (fft_energy, fft_entropy, low/high frequency power). Ils communiquent une information sur la **répartition spectrale** des détails.
        - **Descripteurs de gradient** : Les descripteurs HOG (moyenne, écart-type, entropie) résument les **orientations dominantes** des gradients, utiles pour capturer les structures visuelles.
        """)

        st.markdown("""
        Ces descripteurs sont concaténés pour former un **vecteur unique par image**, servant ensuite d’entrée aux algorithmes de classification. 
        Les caractéristiques générées sont consignées dans un tableau (annexe 6.5), précisant pour chaque descripteur : sa source ou librairie d’origine, ainsi que sa fonction ou utilité dans le cadre de notre projet.

        Certaines variables sont directement issues du traitement d’image (par exemple via OpenCV, NumPy, skimage.feature ou la transformée de Fourier), 
        tandis que d’autres jouent un rôle de support (identifiant de la plante, label, cible, ou métadonnée de structure). Cette organisation facilite l’analyse, la traçabilité, ainsi que la future sélection des features les plus discriminantes pour la phase de classification.
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
        Nous avons réparti le travail en équipe avec un modèle par membre (**SVM, XGBoost, Extra-Trees, Régression Logistique**).
        Pour permettre une comparaison équitable, nous présentons ici uniquement les résultats de l'**Objectif 1 : identification de l'espèce**.
        """)

        with st.expander("Méthodologie d'Évaluation Commune"):
            st.markdown("""
            - **Dataset** : PlantVillage segmented & nettoyé.
            - **Pré-traitement** : Standardisation (RobustScaler).
            - **Validation** : Cross-Validation Stratifiée (5-fold) sur Train.
            - **Métriques** : Accuracy, Précision, Rappel, F1-score (Macro avg pour gérer le déséquilibre).
            """)

        # Données complètes
        full_perf_data = {
            "Modèle": ["SVM (RBF)", "XGBoost", "Reg. Logistique", "Extra-Trees"],
            "Accuracy": [0.9370, 0.9038, 0.8615, 0.8310],
            "Précision (macro)": [0.9271, 0.9051, 0.8462, 0.8607],
            "Rappel (macro)": [0.9207, 0.8654, 0.8214, 0.7405],
            "F1-score (macro)": [0.9237, 0.8839, 0.8328, 0.7863]
        }
        df_full = pd.DataFrame(full_perf_data)

        # Graphique et Tableau
        st.subheader("Performances Globales sur le Test Set")
        
        col_tab, col_chart = st.columns([1, 1.5])
        with col_tab:
            # Fix: Apply format/highlight only to numeric columns to avoid error on 'Modèle' column
            numeric_cols = ["Accuracy", "Précision (macro)", "Rappel (macro)", "F1-score (macro)"]
            st.dataframe(df_full.style.highlight_max(axis=0, color='lightgreen', subset=numeric_cols).format("{:.4f}", subset=numeric_cols))
            
            st.markdown("""
            **Interprétation :**
            - **SVM-RBF** démontre une excellente capacité de généralisation (>93% accuracy) et un équilibre Précision/Rappel robuste.
            - **XGBoost** arrive en seconde position (~4 points derrière), avec un rappel légèrement plus faible sur les classes minoritaires.
            - **Extra-Trees** affiche les résultats les plus faibles, peinant à reconnaître correctement l'ensemble des espèces (Rappel ~0.74).
            """)
        
        with col_chart:
            df_melt = df_full.melt(id_vars="Modèle", var_name="Métrique", value_name="Valeur")
            fig_full = px.bar(df_melt, x="Modèle", y="Valeur", color="Métrique", barmode="group",
                            text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_full.update_layout(yaxis_range=[0.7, 1.0], margin=dict(t=0, b=0, l=0, r=0), showlegend=True, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_full, use_container_width=True)



        st.divider()
        st.subheader("Analyse détaillée par Modèle & Espèce")
        st.markdown("""
        Les espèces les mieux reconnues sont **Blueberry**, **Tomato** et **Grape** (support important).
        À l'inverse, **Pepper_bell**, **Potato** et **Strawberry** posent des défis considérables (ambiguïté morphologique).
        Le **SVM-RBF** gère mieux ces classes difficiles grâce à son noyau non-linéaire et ses poids de classes ajustés.
        """)

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
        
        # Display in 4 columns
        cols = st.columns(4)
        
        # Define class mapping based on consistent support values across reports
        class_mapping = {
            "0": "Apple", "1": "Blueberry", "2": "Cherry_(sour)", 
            "3": "Corn_(maize)", "4": "Grape", "5": "Orange", 
            "6": "Peach", "7": "Pepper,_bell", "8": "Potato", 
            "9": "Raspberry", "10": "Soybean", "11": "Squash", 
            "12": "Strawberry", "13": "Tomato"
        }

        for i, (model_name, paths) in enumerate(model_paths.items()):
            with cols[i]:
                st.markdown(f"**{model_name}**")
                
                # Dynamic transform origin based on column index to prevent clipping
                if i == 0:
                    t_origin = "left top"
                elif i == 3:
                    t_origin = "right top"
                else:
                    t_origin = "center top"

                # Report Table as Image
                if os.path.exists(paths["report"]):
                    df_report = parse_classification_report(paths["report"])
                    if df_report is not None:
                        # Fix for XGBoost which uses numeric class labels
                        if df_report["Classe"].iloc[0] in ["0", 0]: 
                            df_report["Classe"] = df_report["Classe"].astype(str).map(class_mapping).fillna(df_report["Classe"])

                        # Simplify for display
                        df_display = df_report.copy()
                        df_display["Précision"] = df_display["Précision"].apply(lambda x: f"{x:.2f}")
                        df_display["Rappel"] = df_display["Rappel"].apply(lambda x: f"{x:.2f}")
                        df_display["F1-score"] = df_display["F1-score"].apply(lambda x: f"{x:.2f}")
                        
                        img_buf = render_df_as_image(df_display)
                        if isinstance(img_buf, (io.BytesIO, bytes)):
                            # Convert to base64 for HTML embedding with custom class
                            b64 = base64.b64encode(img_buf.getvalue()).decode()
                            html = f'<img src="data:image/png;base64,{b64}" width="200" class="perf-zoom" style="transform-origin: {t_origin};" alt="Rapport">'
                            st.markdown(html, unsafe_allow_html=True)
                        else:
                            st.error(f"Erreur: {img_buf}")
                
                # Confusion Matrix
                if os.path.exists(paths["plot"]):
                    with open(paths["plot"], "rb") as f:
                        b64_cm = base64.b64encode(f.read()).decode()
                    html_cm = f'<img src="data:image/png;base64,{b64_cm}" width="200" class="perf-zoom" style="transform-origin: {t_origin};" alt="Matrice">'
                    st.markdown(html_cm, unsafe_allow_html=True)
                else:
                    st.info("CM non disponible")
                
    with tabs[2]:
        st.header("3. Interprétabilité SHAP")
        
        st.markdown("""
        **Analyse de l'importance globale :**
        Le graphique ci-dessous présente le **Top 10** des caractéristiques les plus influentes. La **densité de contour** (*contour_density*) se détache nettement, confirmant que la morphologie des zones impactées est un indicateur majeur. Elle est suivie par les descripteurs de **couleur** (*mean_R, mean_B*) et de **fréquence** (*fft_entropy*), validant notre approche multi-facettes (forme, teinte et détails fins).

        **Zoom sur les spécificités par classe :**
        L'analyse SHAP révèle que chaque pathologie a sa propre "signature visuelle". Par exemple, la texture locale (*hog_std*) est déterminante pour identifier l'**Apple_scab**, tandis que la densité de contour reste une feature robuste pour l'ensemble des espèces.

        **Synthèse des apprentissages :**
        - **Complémentarité** : Aucune catégorie (couleur, texture, forme) ne suffit seule ; c'est leur combinaison qui assure la performance.
        - **Diversité** : Les 34 features extraites contribuent toutes significativement à la décision finale du modèle.
        - **Pertinence** : Le modèle s'appuie sur des critères visuels cohérents avec une expertise agronomique (couleur des taches, relief des lésions).
        """)
        
        shap_report_path = "figures/shap_analysis/shap_analysis_report.json"
        
        if os.path.exists(shap_report_path):
             with open(shap_report_path, 'r') as f:
                shap_data = json.load(f)
            
             if "top_10_features_globales" in shap_data:
                top_features = shap_data["top_10_features_globales"]
                df_shap = pd.DataFrame(list(top_features.items()), columns=["Feature", "SHAP Value"])
                df_shap = df_shap.sort_values(by="SHAP Value", ascending=True)

                fig_shap = px.bar(
                    df_shap, 
                    x="SHAP Value", 
                    y="Feature", 
                    orientation='h',
                    title="Top 10 Features (Impact moyen sur la prédiction)",
                    color="SHAP Value",
                    color_continuous_scale="Viridis"
                )
                fig_shap.update_layout(showlegend=False)
                st.plotly_chart(fig_shap, use_container_width=True)
             else:
                st.warning("Données 'top_10_features_globales' introuvables dans le rapport JSON.")
        else:
             st.warning(f"Fichier de données SHAP introuvable : {shap_report_path}")




def sidebar_choice():
    st.title("Machine Learning")
    render_ml_content()

