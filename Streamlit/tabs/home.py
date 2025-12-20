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

    # --- PROJECT CONTEXT ---
    st.markdown("## 🌍 Contexte & Enjeux")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
        <h3>🚨 Le Problème</h3>
        <p>Les maladies des plantes menacent la sécurité alimentaire mondiale (pertes estimées à plusieurs centaines de milliards de dollars par la FAO). 
        Une détection précoce est cruciale mais difficile sans expertise onéreuse.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class='card'>
        <h3>🎯 L'Objectif Global</h3>
        <p>Concevoir une solution basée sur l'IA pour identifier automatiquement les espèces végétales, évaluer leur santé et diagnostiquer les maladies.</p>
        </div>
        """, unsafe_allow_html=True)

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

    # --- GALLERY ---
    st.subheader("📸 Aperçu du Dataset PlantVillage")
    
    image_dirs = [
        "Deep_Learning/Interpretability/gradcam_input/specie_background_changed/",
        "Deep_Learning/Interpretability/gradcam_input/in_wild/"
    ]
    
    found_images = []
    for d in image_dirs:
        if os.path.exists(d):
            imgs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.png', '.jpg'))]
            found_images.extend(imgs)
            
    if found_images:
        cols_img = st.columns(4)
        for i, img_path in enumerate(found_images[:4]):
             with cols_img[i]:
                caption = os.path.basename(img_path).split("___")[0] 
                if len(caption) > 15: caption = caption[:15] + "..."
                st.image(img_path, caption=caption, use_container_width=True)

