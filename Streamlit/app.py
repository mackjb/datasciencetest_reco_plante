import os
import streamlit as st
from tabs import home, eda, modeling, machine_learning, deep_learning, proof_of_concept, conclusion

st.set_page_config(
    page_title="Reco-Plante",
    page_icon="🌿",
    layout="wide"
)

# Injection de CSS pour un look "Premium"
st.markdown("""
<style>
    /* Width of the sidebar */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#2e7d32, #1b5e20);
        color: white;
        width: 350px !important;
        min-width: 350px !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
        font-weight: bold;
        font-size: 1.1rem !important;
    }
    /* Increase font size for navigation items */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        font-size: 1.2rem !important;
        color: white !important;
    }
    /* Sidebar title styling */
    [data-testid="stSidebar"] h1 {
        font-size: 2rem !important;
        color: white !important;
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
    # Icône en haut de la navigation
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 4rem;'>🌿</h1>", unsafe_allow_html=True)
    
    
    
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
    selection = st.sidebar.radio("Aller à", list(pages.keys()), label_visibility="collapsed")

    logo_path = "Streamlit/assets/logo_datascientest.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=250)
    
    # Appel de la fonction de la page sélectionnée
    page = pages[selection]
    page.sidebar_choice()

if __name__ == "__main__":
    main()
