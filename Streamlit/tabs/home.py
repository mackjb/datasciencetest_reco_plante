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
        margin-bottom: 5px;
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

    st.markdown("""
    <div class='sub-text'>
    <b>Projet de Reconnaissance des Plantes et leur Maladies par Vision par Ordinateur</b><br>
    Certification Data Scientist - Promotion Mars 2025
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col_card_left, col_illustration, col_card_right = st.columns([5, 1, 5])

    with col_card_left:
        st.info("#### Contexte\n\n"
                "Les plantes occupent une place essentielle dans notre vie quotidienne, que ce soit pour alimenter les populations, "
                "protéger la biodiversité ou embellir nos paysages. Pourtant, leur santé est régulièrement menacée par des maladies, des "
                "parasites ou des conditions climatiques défavorables, entraînant des pertes économiques colossales — estimées à plusieurs "
                "centaines de milliards de dollars par an selon la FAO — et compromettant la sécurité alimentaire mondiale. En parallèle, "
                "l’engouement du grand public pour la reconnaissance des espèces et la compréhension de leur état de santé ne cesse de croître, "
                "comme le montre l’adoption massive d’outils collaboratifs tels que PlantNet.")

    with col_illustration:
        st.markdown(
            """
            <div style="display:flex; justify-content:center; align-items:center; height:100%; min-height:120px;">
                <span style="font-size:64px; color:#555; font-weight:700;">→</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_card_right:
        st.success("#### Mission\n\n"
                   "Ce projet s’inscrit dans cette dynamique, développé dans le cadre d’une formation en data science. "
                   "Il consiste à concevoir une solution basée sur des techniques d’intelligence artificielle pour identifier automatiquement "
                   "les espèces végétales, évaluer leur santé et diagnostiquer d’éventuelles maladies. "
                   "À travers ce travail, nous mettons en pratique des méthodes avancées d’apprentissage automatique tout en nous confrontant à des enjeux réels.")

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
    Notre projet s’inspire d’[une revue systématique de 2024](https://www.researchgate.net/publication/384455442_A_systematic_review_of_deep_learning_techniques_for_plant_diseases), qui confirme la supériorité du Deep Learning pour la reconnaissance 
    des plantes. Nous avons cependant implémenté une baseline en Machine Learning classique, principalement pour appliquer les 
    méthodes enseignées dans le cursus Data Scientist.
    Cette approche ne vise pas à rivaliser avec le Deep Learning, que nous avons largement exploré et optimisé à travers différentes 
    techniques pour en améliorer significativement les performances.
    """
        )

    # --- L'ÉQUIPE PROJET ---
    st.markdown("## L'Équipe Projet")
    c_left, c_center, c_right = st.columns([15, 70, 15])
    with c_center:
        st.image("Streamlit/assets/equipe.png", use_container_width=True)

    st.divider()

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



    
