import streamlit as st
import pandas as pd
import os
import plotly.express as px

def sidebar_choice():
    st.title("🔎 Analyse Exploratoire & Preprocessing")
    
    tab1, tab2, tab3 = st.tabs(["📊 Le Dataset", "🧹 Nettoyage", "📈 Visualisation"])
    
    with tab1:
        st.header("1. Le Dataset PlantVillage")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
            **Source** : Le dataset **PlantVillage** (version Segmented).
            
            *   **Volumétrie** : 54,306 images.
            *   **Espèces** : 14 espèces (Tomate, Pomme, Maïs, etc.).
            *   **Classes** : 38 (combinaisons espèce-maladie-sain).
            *   **Qualité** : Images de feuilles cadrées sur fond uni.
            """)
            
            st.info("🎯 **Objectif** : Utiliser ces images pour l'identification de l'espèce, l'état de santé et le diagnostic du type de maladie.")

        with c2:
            st.metric("Nombre d'images", "54,306")
            st.metric("Espèces", "14")
            st.metric("Classes", "38")
            
    with tab2:
        st.header("2. Pipeline de Preprocessing")
        st.markdown("""
        Pour garantir la robustesse du modèle lors du passage en production (images réelles), nous avons appliqué un nettoyage strict.
        """)
        
        st.markdown("### 🛠 Étapes Clés du Nettoyage")
        st.markdown("""
        1.  **Suppression des Images Inexploitables** : 18 images détectées comme presque noires ont été retirées.
        2.  **Détection de Doublons** : 21 doublons d'images ont été supprimés pour éviter tout biais.
        3.  **Redimensionnement** : Uniformisation de toutes les images en **256 x 256 pixels**.
        """)
        
        st.divider()
        st.markdown("### 🧬 Catégories de Caractéristiques Extraites")
        st.markdown("""
        Pour le Machine Learning classique, nous avons extrait :
        *   **Morphologie** : Aire, périmètre, circularité, excentricité.
        *   **Colorimétrie** : Moyennes & écarts-types RGB/HSV.
        *   **Texture** : Haralick (GLCM), Netteté, Contrastes.
        *   **Fréquentiel** : Transformée de Fourier (FFT), entropie spectrale.
        *   **Descripteurs** : Moments de Hu (invariants), HOG (gradients).
        """)
        
    with tab3:
        st.header("3. Visualisation des Données")
        st.write("Exploration de la distribution des classes.")
        
        # Chargement des données réelles
        cnt_path = "results/Deep_Learning/archi1_outputs_mono_disease_effv2s_256_color_split/class_counts.csv"
        
        if os.path.exists(cnt_path):
            df_counts = pd.read_csv(cnt_path)
            # Nettoyage des noms de classes pour l'affichage
            df_counts['class_name'] = df_counts['class'].apply(lambda x: x.replace("___", " - ").replace("_", " ").title())
            
            fig = px.bar(df_counts, x='count', y='class_name', orientation='h', 
                         title="Distribution du nombre d'images par Classe",
                         labels={'count': "Nombre d'images", 'class_name': "Classe"},
                         color='count', color_continuous_scale='Viridis')
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.metric("Total Images (Train)", df_counts['count'].sum())
            
        else:
            st.warning(f"⚠️ Fichier de données introuvable : {cnt_path}")
            
        st.markdown("### Exemple : Sain vs Malade")
        
        # Tentative de recherche d'exemplaires dans les dossiers de résultats/gradcam
        base_dir = "Deep_Learning/Interpretability/gradcam_input/specie_background_changed/"
        if os.path.exists(base_dir):
            imgs = sorted([f for f in os.listdir(base_dir) if f.endswith(".png")])
            if len(imgs) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(os.path.join(base_dir, imgs[0]), caption="Exemple A", use_container_width=True)
                with c2:
                    st.image(os.path.join(base_dir, imgs[1]), caption="Exemple B", use_container_width=True)


