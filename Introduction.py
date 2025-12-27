import streamlit as st


def main():
    st.set_page_config(
        page_title="RECONNAISSANCE DES PLANTES",
        page_icon="🌱",
        layout="wide",
    )

    st.title("RECONNAISSANCE DES PLANTES")
    st.subheader("Application Streamlit")

    st.markdown(
        """
        Cette application présente :

        - l'exploration des données (EDA) et le pré-traitement,
        - la modélisation **Machine Learning** (features engineering) et **Deep Learning**,
        - deux cas d'étude en Deep Learning (Cas X et Cas Y),
        - une synthèse des **conclusions & perspectives**,
        - une page **About** avec le contexte du projet.

        Utilisez le menu de navigation (barre latérale) pour parcourir les différentes pages.
        """
    )


if __name__ == "__main__":
    main()