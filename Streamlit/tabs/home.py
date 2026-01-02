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
        background-color: #a2d2ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    <h1 class='main-header'> DataScienceTest : Reco Plante</h1>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])

    st.markdown("""
    <div class='sub-text'>
    <b>Projet de Reconnaissance des Plantes et leur Maladies par Vision par Ordinateur</b><br>
    Certification Data Scientist - Promotion Mars 2025
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col_card_left, col_illustration, col_card_right = st.columns(3)

    with col_card_left:
        st.markdown(
            """
            <div class="card" style="background-color: #a2d2ff;">
                <h4>Contexte</h4>
                <p>
                "Les plantes occupent une place essentielle dans notre vie quotidienne, que ce soit pour alimenter les populations, 
                 protéger la biodiversité ou embellir nos paysages. Pourtant, leur santé est régulièrement menacée par des maladies, des 
                 parasites ou des conditions climatiques défavorables, entraînant des pertes économiques colossales — estimées à plusieurs 
                 centaines de milliards de dollars par an selon la FAO — et compromettant la sécurité alimentaire mondiale. En parallèle, 
                 l’engouement du grand public pour la reconnaissance des espèces et la compréhension de leur état de santé ne cesse de croître, 
                 comme le montre l’adoption massive d’outils collaboratifs tels que PlantNet."
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_illustration:
        st.markdown(
            """
            <div class="card" style="background-color: #ffffff; text-align:center;">
            """,
            unsafe_allow_html=True,
        )

        st.image(
            "Streamlit/assets/implique.png",
            width=80,
        )

        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_card_right:
        st.markdown(
            """
            <div class="card" style="background-color: #caffbf;">
                <h4>Mission</h4>
                <p>
                "Ce projet s’inscrit dans cette dynamique, développé dans le cadre d’une formation en data science."
                "Il consiste à concevoir une solution basée sur des techniques d’intelligence artificielle pour identifier automatiquement" 
                "les espèces végétales, évaluer leur santé et diagnostiquer d’éventuelles maladies."
                "À travers ce travail, nous mettons en pratique des méthodes avancées d’apprentissage automatique tout en nous confrontant à des enjeux réels." 
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- OBJECTIFS SPECIFIQUES ---
    st.markdown("## Objectifs du Projet")
    obj1, obj2, obj3 = st.columns(3)
    with obj1:
        st.info("**1. Classification d'espèce**\n\nQuelle est cette plante ? (14 espèces cibles)")
    with obj2:
        st.success("**2. État de santé**\n\nLa plante est-elle saine ou malade ?")
    with obj3:
        st.warning("**3. Diagnostic Maladie**\n\nQuelle est la maladie spécifique ? (20 classes)")

    # --- L'ÉQUIPE PROJET ---
    col_team, col_env = st.columns(2)
    with col_team:
        st.markdown("## L'Équipe Projet")
        st.image("Streamlit/assets/equipe.png", width=600)

    # --- ENVIRONNEMENT DE DÉVELOPPEMENT ---
    with col_env:
        st.markdown("## Environnement de développement")
        st.image("Streamlit/assets/env_dev.png", width=800)



    # --- DECORATIVE GALLERY ---
    st.markdown("""
    <style>
    .leaf-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 350px;
        position: relative;
        margin: 20px 0;
        overflow: visible;
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
        transform: scale(2.0) rotate(0deg) !important;
        z-index: 1000 !important;
        border-color: #2E8B57;
        box-shadow: 0 15px 35px rgba(0,0,0,0.8);
    }
    </style>
    """, unsafe_allow_html=True)


    
