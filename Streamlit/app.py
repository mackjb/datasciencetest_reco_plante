import os
import streamlit as st
from tabs import home, eda, modeling, dl_exploration, proof_of_concept, conclusion, ml_roadmap

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
        background-image: linear-gradient(#43a047, #2e7d32);
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
    
    /* Reduce top padding for main container */
    .block-container {
        padding-top: 2rem !important;
        margin-top: -1rem !important;
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
        "Machine Learning": ml_roadmap,
        "Deep Learning": dl_exploration,
        "PoCs": proof_of_concept,
        "Conclusion & Perspectives": conclusion,
    }
    
    # Création du menu radios
    selection = st.sidebar.radio("Aller à", list(pages.keys()), label_visibility="collapsed")

    st.sidebar.markdown(
        "<div style='text-align: center; margin-top: 10px; margin-bottom: 10px;'>"
        "<a href='https://github.com/mackjb/datasciencetest_reco_plante' target='_blank' rel='noopener noreferrer' "
        "style='display: inline-flex; align-items: center; justify-content: center; text-decoration: none;'>"
        "<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 16 16' fill='white' aria-label='GitHub'>"
        "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z'/></svg>"
        "</a>"
        "</div>",
        unsafe_allow_html=True,
    )

    logo_path = "Streamlit/assets/logo_datascientest.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=250)
    
    # Appel de la fonction de la page sélectionnée
    page = pages[selection]
    page.sidebar_choice()

if __name__ == "__main__":
    main()
