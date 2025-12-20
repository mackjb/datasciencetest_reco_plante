import streamlit as st
import pandas as pd
import os
import plotly.express as px

def sidebar_choice():
    st.title("🧠 Deep Learning : Approche CNN")
    
    st.markdown("""
    Nous avons suivi une démarche structurée en explorant **9 architectures** différentes pour identifier le meilleur compromis entre précision, latence et maintenabilité.
    
    **Backbone commun** : EfficientNetV2S (pré-entraîné sur ImageNet).
    """)
    
    tab_archis, tab_results, tab_demo = st.tabs(["🏗️ Les 9 Architectures", "📊 Sélection & Résultats", "🔮 Interprétabilité (Grad-CAM)"])
    
    with tab_archis:
        st.header("Exploration des 9 Architectures")
        st.markdown("""
        Les architectures sont réparties en deux groupes principaux :
        1.  **Backbone dédié** : Un réseau complet pour chaque objectif (Espèce, Santé, Maladie).
        2.  **Backbone partagé** : Un seul réseau avec plusieurs têtes de sortie (Multi-tâches).
        """)
        
        with st.expander("Détails des architectures 1 à 9"):
            st.markdown("""
            *   **Archi 1** : 3 modèles indépendants (Spécialisation maximale).
            *   **Archi 2** : 2 modèles (Espèce + Santé/Maladie combinées).
            *   **Archi 3** : 1 modèle / 1 tête (35 classes combinées - Idéal Mobile).
            *   **Archi 4** : Architecture en CASCADE (Espèce -> Maladie).
            *   **Archi 5** : CNN + SVM (Hybride DL/ML).
            *   **Archi 6** : Multi-tâche unifiée (Sans fine-tuning).
            *   **Archi 7** : Multi-tâche à 2 têtes (Espèce + Maladie masquée).
            *   **Archi 8** : Multi-tâche simplifiée.
            *   **Archi 9** : Architecture conditionnée (Espèce + Santé -> Maladie - Notre Choix Production).
            """)
            
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Generic_Feed_forward_neural_network.svg/400px-Generic_Feed_forward_neural_network.svg.png", caption="Concept de Backbone partagé (Multi-task Learning)", width=400)

    with tab_results:
        st.header("Sélection des Meilleurs Modèles")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.success("🏆 **Production Standard : Archi 9**")
            st.markdown("""
            *   **F1-score (macro)** : ~99.55%
            *   **Avantages** : Précision maximale, robustesse via conditionnement hiérarchique.
            *   **Usage** : Cloud, Applications professionnelles.
            """)
            
        with col_res2:
            st.info("📱 **Mobile / Edge : Archi 3**")
            st.markdown("""
            *   **F1-score (macro)** : ~99.53%
            *   **Avantages** : Simplicité (1 modèle), latence minimale.
            *   **Usage** : Smartphones, embarqué.
            """)
            
        st.divider()
        st.subheader("Synthèse des Performances")
        st.markdown("Comparaison Accuracy vs F1-Score pour le diagnostic complet.")
        
        # Données de synthèse du rapport
        arch_data = {
            "Architecture": ["Archi 9", "Archi 7", "Archi 1", "Archi 3", "Archi 2", "Archi 5"],
            "Macro F1-Score": [0.9955, 0.9951, 0.9950, 0.9953, 0.9912, 0.9821],
            "Accuracy": [0.9970, 0.9968, 0.9968, 0.9972, 0.9955, 0.9910]
        }
        df_arch = pd.DataFrame(arch_data)
        st.plotly_chart(px.bar(df_arch, x="Architecture", y=["Macro F1-Score", "Accuracy"], barmode="group", color_discrete_sequence=["#2E7D32", "#81C784"]), use_container_width=True)

    # --- DEMO ---
    with tab_demo:
        st.header("Interprétabilité (Grad-CAM)")
        st.markdown("""
        L'interprétabilité permet de valider que le modèle base sa décision sur des **lésions réelles** et non sur des biais (fond, lumière).
        """)
        
        st.subheader("Pertinence des Prédictions")
        
        # Galerie Grad-CAM
        gradcam_dir = "Deep_Learning/Interpretability/gradcam_input/specie_background_changed/"
        if os.path.exists(gradcam_dir):
            imgs = [os.path.join(gradcam_dir, f) for f in os.listdir(gradcam_dir) if f.endswith(".png")]
            if imgs:
                st.image(imgs[0], caption="Exemple d'activation Grad-CAM", width=400)
                if len(imgs) > 1:
                     with st.expander("Voir plus d'exemples"):
                         st.image(imgs[1:4], width=200)
        else:
            st.warning("Images Grad-CAM non trouvées.")

