import streamlit as st
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
from tabs import home, eda, modeling, machine_learning, deep_learning, proof_of_concept, conclusion

st.set_page_config(
    page_title="HerbI-Dent - Reconnaissance des Plantes & Maladies",
    page_icon="🌿",
    layout="wide"
)

# Injection de CSS pour un look "Premium"
st.markdown("""
<style>
    /* Width of the sidebar */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#97DFC6);
        color: white;
        width: 350px !important;
        min-width: 350px !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: black !important;
        font-weight: bold;
        font-size: 1.1rem !important;
    }
    /* Increase font size for navigation items */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        font-size: 1.2rem !important;
        color: black !important;
    }
    /* Sidebar title styling */
    [data-testid="stSidebar"] h1 {
        font-size: 2rem !important;
        color: white !important;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        margin-bottom: 20px;
    }
    /* Modern Card styling */
    .stMetric {
        background-color: #BBDEFB; /* bleu  */
        padding: 2px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.0rem !important;   /* valeur de la métrique */
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;  /* texte 'Poids total', 'Images', etc. */
    }    
    /* Main titles */
    h1 {
        color: #2e7d32;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 10px;
    }
    /* Move page content (and titles) closer to the top */
    .block-container {
        padding-top: 1rem !important;
    }
    h2 {
        color: #388e3c;
    }
    /* Better divider */
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    
    # Dictionnaire des pages
    pages = {
        "Le projet": home,
        "Les jeux de données": eda,
        "Méthodologie ML- DL": modeling,
        "Machine Learning" : machine_learning,
        "Deep Learning" : deep_learning,
        "PoCs": proof_of_concept,
        "Conclusion & Perspectives": conclusion,
    }
    
    # Création du menu radios
    selection = st.sidebar.radio("Aller à", list(pages.keys()))
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.image(os.path.join(ASSETS_DIR, "logo-2021.png"), use_container_width=True)

    # Appel de la fonction de la page sélectionnée
    page = pages[selection]
    page.sidebar_choice()

if __name__ == "__main__":
    main()
