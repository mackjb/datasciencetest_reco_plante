import streamlit as st
import os

def sidebar_choice():
    # --- HEADER ---
    st.markdown("""
    <style>
    .main-header {
        text-align: center; 
        color: #2E8B57;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sub-text {
        text-align: center; 
        font-size: 1.2em; 
        color: #555;
        margin-bottom: 30px;
    }
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    <h1 class='main-header'>🌿 DataScienceTest : Reco Plante</h1>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://www.datasciencetest.com/wp-content/uploads/2020/12/Logo-DataScienceTest-1.png", use_container_width=True)

    st.markdown("""
    <div class='sub-text'>
    <b>Projet de Reconnaissance des Plantes et leur Maladies par Vision par Ordinateur</b><br>
    Certification Data Scientist - Promotion Février 2025
    </div>
    """, unsafe_allow_html=True)

    st.divider()



    # --- OBJECTIFS SPECIFIQUES ---
    st.markdown("## 📋 Objectifs du Projet")
    obj1, obj2, obj3 = st.columns(3)
    with obj1:
        st.info("**1. Classification d'espèce**\n\nQuelle est cette plante ? (14 espèces cibles)")
    with obj2:
        st.success("**2. État de santé**\n\nLa plante est-elle saine ou malade ?")
    with obj3:
        st.warning("**3. Diagnostic précis**\n\nQuelle est la maladie spécifique ? (20 classes)")

    # --- TEAM SECTION ---
    st.markdown("## 👨‍💻 L'Équipe Projet")
    
    if os.path.exists("Streamlit/assets/team_collage.png"):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("Streamlit/assets/team_collage.png", caption="L'équipe au complet", use_container_width=True)
    else:
        st.info("Photo d'équipe à intégrer ici.")

    st.divider()

    # --- DECORATIVE GALLERY ---
    st.markdown("""
    <style>
    .leaf-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 350px; /* Taille diminuée */
        position: relative;
        margin: 20px 0;
        overflow: visible; /* Pour que le zoom ne soit pas coupé */
        background-color: #111;
        border-radius: 20px;
        box-shadow: inset 0 0 30px rgba(0,0,0,0.5);
    }
    .leaf-img {
        position: absolute;
        width: 150px;
        height: 150px;
        object-fit: cover;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
        border: 2px solid #333;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), z-index 0.3s ease;
        background-color: #000;
        cursor: pointer;
    }
    .leaf-img:hover {
        transform: scale(2.0) rotate(0deg) !important; /* Zoom agrandi */
        z-index: 1000 !important;
        border-color: #2E8B57;
        box-shadow: 0 15px 35px rgba(0,0,0,0.8);
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("📸 Immersion dans le Dataset")
    
    img_root = "Deep_Learning/Interpretability/gradcam_input/"
    if os.path.exists(img_root):
        all_imgs = []
        for root, dirs, files in os.walk(img_root):
            for file in files:
                if file.endswith(".png"):
                    all_imgs.append(os.path.join(root, file))
        
        # On essaie d'en avoir une dizaine
        count = min(len(all_imgs), 12)
        if count > 0:
            import base64
            import random
            
            def get_base64(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            
            html_code = "<div class='leaf-container'>"
            # On génère les positions mélangées
            for i in range(count):
                b64 = get_base64(all_imgs[i])
                rot = random.randint(-25, 25)
                tx = (i - (count/2)) * 70 # Espacement horizontal dynamique
                ty = random.randint(-40, 40) # Léger décalage vertical
                z = i
                style = f"transform: rotate({rot}deg) translateX({tx}px) translateY({ty}px); z-index: {z};"
                html_code += f"<img src='data:image/png;base64,{b64}' class='leaf-img' style='{style}'>"
            html_code += "</div>"
            st.markdown(html_code, unsafe_allow_html=True)
        else:
            st.info("Chargement des images...")
    st.divider()

