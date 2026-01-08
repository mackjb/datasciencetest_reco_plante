import streamlit as st
import time

# =========================
# FONCTION PRINCIPALE
# =========================
def render_dl_page():
    # --- CSS ---
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
    /* Cards (alignées sur la page Conclusion) */
    .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        background: rgba(255,255,255,0.03);
        margin: 0.25rem 0 0.9rem 0;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
    }
    [data-theme="light"] .card {
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(0,0,0,0.02);
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
    }
    .card__title {
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }
    .card__body {
        color: rgba(255,255,255,0.82);
        font-size: 0.95rem;
        line-height: 1.35;
    }
    [data-theme="light"] .card__body { color: rgba(0,0,0,0.74); }
    .card--success { border-color: rgba(46, 204, 113, 0.35); }
    .card--warning { border-color: rgba(241, 196, 15, 0.40); }
    .card--info    { border-color: rgba(52, 152, 219, 0.40); }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Architectures Sélectionnées")
    tab_archi3, tab_archi9 = st.tabs([
        "Archi 3 : Solution Edge/Mobile",
        "Archi 9 : Solution Production Cloud"
    ])

    # =========================
    # ARCHI 3
    # =========================
    with tab_archi3:
        col_arch3_text, col_arch3_img = st.columns([1, 1])

        with col_arch3_text:
            st.markdown(
            '''
            <div class="card card--info" style="background-color:#FFE4C4;">
              <div class="card__title">Scénario pour un déploiement sur Applications mobiles / Edge computing :</div>
              <div class="card__body" style="color:#0131B4;">Cas 1 : Identification d'espèce uniquement</div>
            </div>
            ''',
            unsafe_allow_html=True,
            )

            st.markdown(
                '''
                <div class="card card--info" style="margin-top:5rem;background-color:#a2d2ff;">
                  <div class="card__body" style="color:#0b090a; font-weight:bold; text-align:center;">
                    Démo Interactive ↓<br>
                    <span style="font-weight:normal; color:#0131B4;">Sélectionnez une image pour simuler l'inférence et visualiser l'attention du modèle (Grad-CAM).</span>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        with col_arch3_img:
            st.image(
                "/home/vscode/worktrees/bga_dl_experiments/Streamlit/assets/architectures/archi_3_bk.png",
                width=500,
            )

        examples_archi3 = [
            {
                "id": "blueberry",
                "label": "Myrtille (Saine)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/specie/Blueberry___healthy_067e7729-ebb3-4824-80dc-9ceda52f47b8___RS_HL_5388.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_Blueberry___healthy_067e7729-ebb3-4824-80dc-9ceda52f47b8___RS_HL_5388_gradcam_overlay.png",
                "species": "Blueberry",
                "disease": "Healthy",
                "conf": "99.7%"
            },
            {
                "id": "cherry",
                "label": "Cerise (Saine)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/specie/Cherry_(including_sour)___healthy_0008f3d3-2f85-4973-be9a-1b520b8b59fc___JR_HL_4092.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_Cherry_(including_sour)___healthy_0008f3d3-2f85-4973-be9a-1b520b8b59fc___JR_HL_4092_gradcam_overlay.png",
                "species": "Cherry",
                "disease": "Healthy",
                "conf": "99.5%"
            },
            {
                "id": "grape",
                "label": "Raisin (Sain)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/specie/Grape___healthy_00e00912-bf75-4cf8-8b7d-ad64b73bea5f___Mt.N.V_HL_6067.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_Grape___healthy_00e00912-bf75-4cf8-8b7d-ad64b73bea5f___Mt.N.V_HL_6067_gradcam_overlay.png",
                "species": "Grape",
                "disease": "Healthy",
                "conf": "99.9%"
            },
            {
                "id": "potato_healthy",
                "label": "Pomme de terre (Saine)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/specie/Potato___healthy_00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL_1864.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_Potato___healthy_00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL_1864_gradcam_overlay.png",
                "species": "Potato",
                "disease": "Healthy",
                "conf": "99.6%"
            },
            {
                "id": "soybean",
                "label": "Soja (Sain)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/specie/Soybean___healthy_0180c2ed-0393-4e26-89a1-d4031175442f___RS_HL_4556.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_Soybean___healthy_0180c2ed-0393-4e26-89a1-d4031175442f___RS_HL_4556_gradcam_overlay.png",
                "species": "Soybean",
                "disease": "Healthy",
                "conf": "98.8%"
            },
            {
                "id": "strawberry",
                "label": "Fraise (Saine)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/specie/Strawberry___healthy_00166615-5e7b-4318-8957-5e50df335ee8___RS_HL_1785.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi9_disease_interpretability_data/archi9_disease_cam_Strawberry___healthy_00166615-5e7b-4318-8957-5e50df335ee8___RS_HL_1785_gradcam_overlay.png",
                "species": "Strawberry",
                "disease": "Healthy",
                "conf": "99.4%"
            }
        ]

        if "selected_idx3" not in st.session_state:
            st.session_state.selected_idx3 = 0
        if "analyzed3" not in st.session_state:
            st.session_state.analyzed3 = False

        # --- Galerie ---
        cols = st.columns(len(examples_archi3))
        for i, ex in enumerate(examples_archi3):
            with cols[i]:
                border = "5px solid #2E8B57" if st.session_state.selected_idx3 == i else "2px solid #ddd"
                st.markdown(f"<div class='demo-img-container' style='border: {border};'>", unsafe_allow_html=True)
                if st.button(f"Sélect. {ex['id']}", key=f"btn3_{ex['id']}"):
                    st.session_state.selected_idx3 = i
                    st.session_state.analyzed3 = False
                    st.rerun()
                st.image(ex["img_orig"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        selected3 = examples_archi3[st.session_state.selected_idx3]

        c1, c2, c3 = st.columns([1, 0.8, 1])
        with c1:
            st.markdown("Entrée")
            st.image(selected3["img_orig"], caption="Image originale", use_container_width=True)

        with c2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if st.button("Lancer l'Analyse Archi 3", type="primary", use_container_width=True, disabled=st.session_state.analyzed3):
                with st.spinner("Inférence en cours..."):
                    time.sleep(1.2)
                    st.session_state.analyzed3 = True

            if st.session_state.analyzed3:
                st.markdown(f"""
                <div style='background-color:#f1f8e9;padding:20px;border-radius:15px;border:1px solid #c5e1a5;text-align:center;'>
                    <h4 style='color:#2e7d32;margin:0;'>Résultats</h4>
                    <hr>
                    <p><b>Espèce</b> : {selected3['species']}</p>
                    <p><b>Maladie</b> : {selected3['disease']}</p>
                    <p style='font-size:1.2em;color:#2e7d32;'><b>Confiance : {selected3['conf']}</b></p>
                </div>
                """, unsafe_allow_html=True)

        with c3:
            st.markdown("Interprétation")
            if st.session_state.analyzed3:
                st.image(selected3["img_cam"], caption="Grad-CAM (zones influentes)", use_container_width=True)
            else:
                st.info("Lancez l'analyse pour visualiser la carte de chaleur.")

    # =========================
    # ARCHI 9 
    # =========================
    with tab_archi9:
        col_arch9_text, col_arch9_img = st.columns([1, 1])

        with col_arch9_text:
            st.markdown(
                '''
                <div class="card card--info" style="background-color:#FFE4C4;">
                  <div class="card__title">Scénario pour déploiement en Production standard – Applications professionnelles</div>
                  <div class="card__body" style="color:#0131B4;">Cas 2 : Diagnostic ciblé (espèce connue → maladie) - Archi 9 : Solution Production Cloud</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            st.markdown(
                '''
                <div class="card card--info" style="margin-top:5rem;background-color:#a2d2ff;">
                  <div class="card__body" style="color:#0b090a; font-weight:bold; text-align:center;">
                    Démo Interactive ↓<br>
                    <span style="font-weight:normal; color:#0131B4;">Sélectionnez une image pour simuler l'inférence et visualiser l'attention du modèle (Grad-CAM).</span>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        with col_arch9_img:
            st.image(
                "/home/vscode/worktrees/bga_dl_experiments/Streamlit/assets/architectures/archi_9_bk.png",
                width=500,
            )

        examples_archi9 = [
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
                "id": "corn_nlb",
                "label": "Maïs (Brûlure)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Corn_(maize)___Northern_Leaf_Blight_0079c731-80f5-4fea-b6a2-4ff23a7ce139___RS_NLB_4121.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Corn_(maize)___Northern_Leaf_Blight_0079c731-80f5-4fea-b6a2-4ff23a7ce139___RS_NLB_4121_gradcam_overlay.png",
                "species": "Corn",
                "disease": "Northern Leaf Blight",
                "conf": "98.9%"
            },
            {
                "id": "corn_cercospora",
                "label": "Maïs (Tache grise)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot_00120a18-ff90-46e4-92fb-2b7a10345bd3___RS_GLSp_9357.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot_00120a18-ff90-46e4-92fb-2b7a10345bd3___RS_GLSp_9357_gradcam_overlay.png",
                "species": "Corn",
                "disease": "Cercospora / Gray Leaf Spot",
                "conf": "97.6%"
            },
            {
                "id": "apple",
                "label": "Pomme (Rouille)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Apple___Cedar_apple_rust_025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust_3655.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Apple___Cedar_apple_rust_025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust_3655_gradcam_overlay.png",
                "species": "Apple",
                "disease": "Cedar Rust",
                "conf": "99.2%"
            },
            {
                "id": "orange",
                "label": "Orange (Greening)",
                "img_orig": "Deep_Learning/Interpretability/gradcam_input/disease/Orange___Haunglongbing_(Citrus_greening)_01c0f6d7-5f35-404e-8d6d-cadc3dfafb59___UF.Citrus_HLB_Lab_0068.JPG",
                "img_cam": "results/Deep_Learning/gradcam_outputs/archi3_disease_interpretability_data/archi3_disease_cam_Orange___Haunglongbing_(Citrus_greening)_01c0f6d7-5f35-404e-8d6d-cadc3dfafb59___UF.Citrus_HLB_Lab_0068_gradcam_overlay.png",
                "species": "Orange",
                "disease": "Huanglongbing (Citrus Greening)",
                "conf": "98.3%"
            }
        ]

        if "selected_idx9" not in st.session_state:
            st.session_state.selected_idx9 = 0
        if "analyzed9" not in st.session_state:
            st.session_state.analyzed9 = False

        cols = st.columns(len(examples_archi9))
        for i, ex in enumerate(examples_archi9):
            with cols[i]:
                border = "5px solid #2E8B57" if st.session_state.selected_idx9 == i else "2px solid #ddd"
                st.markdown(f"<div class='demo-img-container' style='border: {border};'>", unsafe_allow_html=True)
                if st.button(f"Sélect. {ex['id']}", key=f"btn9_{ex['id']}"):
                    st.session_state.selected_idx9 = i
                    st.session_state.analyzed9 = False
                    st.rerun()
                st.image(ex["img_orig"], width=150)
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        selected9 = examples_archi9[st.session_state.selected_idx9]

        c1, c2, c3 = st.columns([1, 0.8, 1])
        with c1:
            st.markdown("Entrée")
            st.image(selected9["img_orig"], caption="Image originale", width=300)

        with c2:
            if st.button("Lancer l'Analyse Archi 9", type="primary", width=300, disabled=st.session_state.analyzed9):
                with st.spinner("Inférence en cours..."):
                    time.sleep(1.2)
                    st.session_state.analyzed9 = True

            if st.session_state.analyzed9:
                st.markdown(f"""
                <div style='background-color:#f1f8e9;padding:20px;border-radius:15px;border:1px solid #c5e1a5;text-align:center;'>
                    <h4 style='color:#2e7d32;margin:0;'>Résultats</h4>
                    <hr>
                    <p><b>Espèce</b> : {selected9['species']}</p>
                    <p><b>Maladie</b> : {selected9['disease']}</p>
                    <p style='font-size:1.2em;color:#2e7d32;'><b>Confiance : {selected9['conf']}</b></p>
                </div>
                """, unsafe_allow_html=True)

        with c3:
            st.markdown("Interprétation")
            if st.session_state.analyzed9:
                st.image(selected9["img_cam"], caption="Grad-CAM (zones influentes)", width=300)
            else:
                st.info("Lancez l'analyse pour visualiser la carte de chaleur.")

# =========================
# SIDEBAR
# =========================
def sidebar_choice():
    st.title("PoCs - Deep Learning")
    render_dl_page()

# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    sidebar_choice()
