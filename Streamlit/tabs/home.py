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
    <h1 class='main-header'>🌿 DataScientest : Reco Plante</h1>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    ''' col1, col2, col3 = st.columns([1,2,1])
    with col2:
        logo_path = "Streamlit/assets/logo_datascientest.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=350) '''

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

    st.markdown(
            """
    Notre projet s’inspire d’une revue systématique de 2024, qui confirme la supériorité du Deep Learning pour la reconnaissance 
    des plantes. Nous avons cependant implémenté une baseline en Machine Learning classique, principalement pour appliquer les 
    méthodes enseignées dans le cursus Data Scientist.
    Cette approche ne vise pas à rivaliser avec le Deep Learning, que nous avons largement exploré et optimisé à travers différentes 
    techniques pour en améliorer significativement les performances.
    """
        )

    # --- L'ÉQUIPE PROJET ---
    col_team, col_env = st.columns(2)
    with col_team:
        st.markdown("## L'Équipe Projet")
        st.image("Streamlit/assets/equipe.png", width=600)

    # --- ENVIRONNEMENT DE DÉVELOPPEMENT ---
    with col_env:
        st.markdown("## Environnement de développement")
        st.image("Streamlit/assets/env_dev.png", width=800)

    st.divider()

    
