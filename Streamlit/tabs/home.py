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


    
