import streamlit as st
import os

from utils import render_mermaid

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
    st.markdown("## L'Équipe Projet")
    st.image("Streamlit/assets/equipe.png", use_container_width=True)

    # --- ENVIRONNEMENT DE DÉVELOPPEMENT ---
    st.markdown("## Environnement de développement")
    
    # Interactive Mindmap Data Definition for Dev Environment
    mindmap_env = {
        "id": "root",
        "text": "Environnements de<br/>développement",
        "children": [
            {
                "id": "Env_Execution",
                "text": "Environnements<br/>d'exécution",
                "children": [
                    {"id": "Codespaces", "text": "GitHub Codespaces reproductible"},
                    {"id": "Local_GPU", "text": "Docker + Machines locales<br/>avec GPU"},
                    {"id": "Compute AML", "text": "Compute AML avec GPU"}
                ]
            },
            {
                "id": "Outils",
                "text": "Outils",
                "children": [
                    {"id": "Windsurf", "text": "Windsurf"},
                    {"id": "VSCode", "text": "Visual Studio Code"},
                    {"id": "Git", "text": "Git"}
                ]
            },
            {
                "id": "Pratique",
                "text": "Pratique de<br/>développement",
                "children": [
                    {"id": "Py", "text": ".py"}
                ]
            },
            {
                "id": "Suivi",
                "text": "Suivi des<br/>expérimentations",
                "children": [
                    {"id": "MLFlow", "text": "MLFlow déployé<br/>sur Azure"}
                ]
            },
            {
                "id": "Gestion_Dep",
                "text": "Gestion des<br/>dépendances",
                "children": [
                    {"id": "Conda", "text": "Conda : fichier env,<br/>Python 3.11"},
                    {
                        "id": "Librairies",
                        "text": "Librairies",
                        "children": [
                            {"id": "Pandas", "text": "pandas"},
                            {"id": "Matplotlib", "text": "matplotlib"},
                            {"id": "Scikit", "text": "scikit-learn"},
                            {"id": "Tensorflow", "text": "TensorFlow, ..."},
                            {"id": "Versions_GPU", "text": "versions adaptées<br/>aux GPUs récents"}
                        ]
                    }
                ]
            }
        ]
    }
    
    render_mermaid(mindmap_env, height=600)

    st.divider()

    
