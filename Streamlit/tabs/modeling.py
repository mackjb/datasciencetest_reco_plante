import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from tabs import deep_learning

def render_approach_content():
    st.markdown("""
    
    Notre projet s’inspire d’une revue systématique de 2024, qui confirme la supériorité du Deep Learning pour la reconnaissance 
    des plantes. Nous avons cependant implémenté une baseline en Machine Learning classique, principalement pour appliquer les 
    méthodes enseignées dans le cursus Data Scientist.
    Cette approche ne vise pas à rivaliser avec le Deep Learning, que nous avons largement exploré et optimisé à travers différentes 
    techniques pour en améliorer significativement les performances.
    """)

    col_ml, col_dl = st.columns(2)

    with col_ml:
        st.markdown("<h3 style='text-align:center;'>Machine Learning</h3>", unsafe_allow_html=True)
        inner_ml = st.columns([1, 2, 1])[1]
        with inner_ml:
            st.image(
                "Streamlit/assets/WF_ML.png",
                use_container_width=True,
            )

    with col_dl:
        st.markdown("<h3 style='text-align:center;'>Deep Learning</h3>", unsafe_allow_html=True)
        inner_dl = st.columns([1, 2, 1])[1]
        with inner_dl:
            st.image(
                "Streamlit/assets/WF_DL.png",
                width=600,
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
