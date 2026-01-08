import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import requests
from tabs import deep_learning

try:
    from streamlit_lottie import st_lottie
except ModuleNotFoundError:
    st_lottie = None

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def render_approach_content():
    st.markdown("""
    <style>
    /* Animation 'Floating Card' pour les images */
    [data-testid="stImage"] img {
        border-radius: 15px; /* Coins arrondis */
        transition: transform 0.3s ease, box-shadow 0.3s ease; /* Transition douce */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Ombre légère de base */
    }
    
    /* Zoom pour l'image de gauche (ML - 2ème colonne) */
    div[data-testid="stColumn"]:nth-of-type(2) [data-testid="stImage"] img:hover {
        transform: scale(1.7); 
        z-index: 1000;
        cursor: zoom-in;
        border-radius: 10px;
        box-shadow: 0 15px 25px rgba(0,0,0,0.25);
    }

    /* Zoom réduit pour l'image de droite (DL - 4ème colonne) */
    div[data-testid="stColumn"]:nth-of-type(4) [data-testid="stImage"] img:hover {
        transform: scale(1.3); /* Plus petit car l'image est déjà grande */
        z-index: 1000;
        cursor: zoom-in;
        border-radius: 10px;
        box-shadow: 0 15px 25px rgba(0,0,0,0.25);
    }
    </style>

    <div style="font-size: 1.2rem; text-align: justify;">
    Notre projet s’inspire d’une revue systématique de 2024, qui confirme la supériorité du Deep Learning pour la reconnaissance 
    des plantes. Nous avons cependant implémenté une baseline en Machine Learning classique, principalement pour appliquer les 
    méthodes enseignées dans le cursus Data Scientist.
    Cette approche ne vise pas à rivaliser avec le Deep Learning, que nous avons largement exploré et optimisé à travers différentes 
    techniques pour en améliorer significativement les performances.
    </div>
    """, unsafe_allow_html=True)
    
    # Animation Lottie (Plante / Recherche)
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json" # Animation plante qui pousse
    lottie_json = load_lottieurl(lottie_url)
    
    # Création de 5 colonnes pour ajouter des marges et réduire la taille des images
    # Ratios : [Marge, ML, Animation, DL, Marge]
    # Rééquilibrage : ML un peu plus petit, DL un peu plus grand
    # Rééquilibrage : Marges plus grandes pour diminuer la taille des images
    _, col_ml, col_anim, col_dl, _ = st.columns([2, 3, 2, 2.5, 2.5])

    with col_ml:
        st.markdown("<h3 style='text-align:center;'>Machine Learning</h3>", unsafe_allow_html=True)
        st.image(
            "Streamlit/assets/WF_ML.png",
            use_container_width=True,
        )

    with col_anim:
        # Centrage vertical approximatif pour l'animation
        st.markdown("<br>" * 3, unsafe_allow_html=True) 
        if lottie_json and st_lottie is not None:
            st_lottie(lottie_json, speed=1, height=150, key="initial") # Taille réduite pour s'intégrer

    with col_dl:
        st.markdown("<h3 style='text-align:center;'>Deep Learning</h3>", unsafe_allow_html=True)
        st.image(
            "Streamlit/assets/WF_DL.png",
            use_container_width=True,
        )

#########################
# CONTENU DEEP LEARNING
#########################
def render_dl_content():
    deep_learning.render_dl_content()

# =========================
# FONCTION PRINCIPALE
# =========================
def sidebar_choice():
    st.title("Méthodologie")
    
    # Affichage direct de l'approche méthodologique
    render_approach_content()
