import streamlit as st
from tabs import home, eda, modeling, deep_learning, conclusion, about

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
        border-bottom: 2px solid rgba(255,255,255,0.3);
        margin-bottom: 20px;
    }
    /* Modern Card styling */
    .stMetric {
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
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
    st.sidebar.title("Navigation")
    
    # Dictionnaire des pages
    pages = {
        "🏠 Accueil": home,
        "🔍 Exploration": eda,
        "📊 Machine Learning": modeling,
        "🧠 Deep Learning": deep_learning,
        "🏁 Bilan & Conclusion": conclusion,
        "👥 L'Equipe": about
    }
    
    # Création du menu radios
    selection = st.sidebar.radio("Aller à", list(pages.keys()))
    
    # Appel de la fonction de la page sélectionnée
    page = pages[selection]
    page.sidebar_choice()

if __name__ == "__main__":
    main()
