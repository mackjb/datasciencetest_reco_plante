import streamlit as st
import pandas as pd
import os
import plotly.express as px

def render_dl_page():
    # --- CUSTOM CSS FOR DEMO ---
    st.markdown("""
    <style>
    .demo-img-container {
        transition: transform 0.2s;
        border-radius: 10px;
        overflow: hidden;
    }
    .demo-img-container:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # --- MAIN TABS ---
    st.subheader("🏗️ Architectures Sélectionnées")
    
    tab_archi3, tab_archi9 = st.tabs(["📱 Archi 3 : Solution Edge/Mobile", "🏆 Archi 9 : Solution Production Cloud"])

    with tab_archi3:
        st.markdown("### 🧬 Démo Interactive : Archi 3 en action")
        st.write("Sélectionnez une image pour simuler une inférence et visualiser l'attention du modèle (Grad-CAM).")

        # Configuration des exemples
        examples = [
            {
                "id": "potato",
                "label": "Pomme de terre (Mildiou)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Potato___Early_blight_001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B_8178.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Potato___Early_blight_001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B_8178_gradcam_overlay.png",
                "species": "Potato",
                "disease": "Early Blight",
                "conf": "99.4%"
            },
            {
                "id": "tomato",
                "label": "Tomate (Virus)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Tomato___Tomato_Yellow_Leaf_Curl_Virus_01f7eeb8-19c7-4c7b-9789-00538abf46fe___UF.GRC_YLCV_Lab_09492.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Tomato___Tomato_Yellow_Leaf_Curl_Virus_01f7eeb8-19c7-4c7b-9789-00538abf46fe___UF.GRC_YLCV_Lab_09492_gradcam_overlay.png",
                "species": "Tomato",
                "disease": "Yellow Leaf Curl Virus",
                "conf": "99.8%"
            },
            {
                "id": "corn",
                "label": "Maïs (Rouille)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Corn_(maize)___Northern_Leaf_Blight_0079c731-80f5-4fea-b6a2-4ff23a7ce139___RS_NLB_4121.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Corn_(maize)___Northern_Leaf_Blight_0079c731-80f5-4fea-b6a2-4ff23a7ce139___RS_NLB_4121_gradcam_overlay.png",
                "species": "Corn",
                "disease": "Northern Leaf Blight",
                "conf": "98.9%"
            },
            {
                "id": "apple",
                "label": "Pomme (Rouille)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Apple___Cedar_apple_rust_025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust_3655.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Apple___Cedar_apple_rust_025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust_3655_gradcam_overlay.png",
                "species": "Apple",
                "disease": "Cedar Rust",
                "conf": "99.2%"
            }
        ]

        if 'selected_idx' not in st.session_state:
            st.session_state.selected_idx = 0

        # --- GALERIE DE SELECTION ---
        cols = st.columns(4)
        for i, ex in enumerate(examples):
            with cols[i]:
                # Style pour l'image sélectionnée
                border = "5px solid #2E8B57" if st.session_state.selected_idx == i else "2px solid #ddd"
                st.markdown(f"<div class='demo-img-container' style='border: {border};'>", unsafe_allow_html=True)
                if st.button(f"Sélect. {ex['id']}", key=f"btn_{ex['id']}"):
                    st.session_state.selected_idx = i
                    st.rerun()
                st.image(ex['img_orig'], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # --- AFFICHAGE RESULTAT ---
        selected = examples[st.session_state.selected_idx]
        
        c1, c2, c3 = st.columns([1, 0.8, 1])
        
        with c1:
            st.markdown("#### 📥 Entrée")
            st.image(selected['img_orig'], caption="Image originale (224x224)", use_container_width=True)
            
        with c2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if st.button("🚀 Lancer l'Analyse Archi 3", type="primary", use_container_width=True):
                with st.spinner("Inférence en cours..."):
                    import time
                    time.sleep(1.2)
                    st.session_state.analyzed = True
            
            if st.session_state.get('analyzed'):
                st.markdown(f"""
                <div style='background-color: #f1f8e9; padding: 20px; border-radius: 15px; border: 1px solid #c5e1a5; text-align: center;'>
                    <h4 style='color: #2e7d32; margin:0;'>Résultats</h4>
                    <hr style='margin: 10px 0;'>
                    <p style='margin: 5px 0;'><b>Espèce</b> : {selected['species']}</p>
                    <p style='margin: 5px 0;'><b>Maladie</b> : {selected['disease']}</p>
                    <p style='font-size: 1.2em; color: #2e7d32; margin-top: 10px;'><b>Confiance : {selected['conf']}</b></p>
                </div>
                """, unsafe_allow_html=True)

        with c3:
            st.markdown("#### 🔍 Interprétation")
            if st.session_state.get('analyzed'):
                st.image(selected['img_cam'], caption="Grad-CAM Overlay (Attention du modèle)", use_container_width=True)
            else:
                st.info("Lancez l'analyse pour visualiser la carte de chaleur.")

    with tab_archi9:
        st.markdown("### Architecture 9 : Modèle Conditionné")
        col_c9_text, col_c9_img = st.columns([1, 1.2])
        
        with col_c9_text:
            st.markdown("""
            **Concept** : Architecture hiérarchique où la tête 'Maladie' est conditionnée par les probabilités d'Espèce et de Santé.
            
            **Points forts :**
            - 🎯 **Précision** : Réduit les confusions entre maladies similaires de différentes plantes.
            - 🧠 **Contextualisation** : Apprend les relations logiques (une maladie spécifique n'affecte que certaines plantes).
            - 🥇 **Choix Final** : Notre meilleur modèle global.
            
            **Performance** : F1-Score record de **99.55%**.
            """)
            
        with col_c9_img:
            st.image("Streamlit/assets/architectures/archi_9.png", caption="Schéma Archi 9 (Hiérarchique)", use_container_width=True)

    st.divider()

    # --- GLOBAL RESULTS & INTERPRETABILITY ---
    st.subheader("📊 Comparaison Finale & Interprétabilité")
    
    col_f1, col_f2 = st.columns([1.2, 1])
    
    with col_f1:
        arch_data = {
            "Architecture": ["Archi 9", "Archi 3", "Archi 7", "Archi 1", "Archi 2", "Archi 5"],
            "F1-Score": [0.9955, 0.9953, 0.9951, 0.9950, 0.9912, 0.9821]
        }
        df_arch = pd.DataFrame(arch_data)
        fig = px.bar(df_arch, x="Architecture", y="F1-Score", color="F1-Score",
                     title="Synthèse des performances (Macro F1)", color_continuous_scale="GnBu")
        fig.update_layout(yaxis_range=[0.97, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with col_f2:
        st.markdown("**Interprétabilité (Grad-CAM)**")
        gradcam_dir = "Deep_Learning/Interpretability/gradcam_input/specie_background_changed/"
        if os.path.exists(gradcam_dir):
            imgs = [f for f in os.listdir(gradcam_dir) if f.endswith(".png")]
            if imgs:
                st.image(os.path.join(gradcam_dir, imgs[0]), caption="Validation visuelle : focus sur les symptômes.", use_container_width=True)
        else:
            st.info("Le modèle Grad-CAM confirme que les décisions sont basées sur les anomalies foliaires.")

def sidebar_choice():
    st.title("🧠 Deep Learning")
    render_dl_page()
