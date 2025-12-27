import streamlit as st


def main():
    st.title("À propos")

    st.markdown(
        """
        Cette page présente :

        - le **contexte** du projet (recommandation de plantes, détection de maladies, etc.),
        - l'auteur / l'équipe,
        - les outils utilisés (TensorFlow/Keras, scikit-learn, etc.),
        - les références éventuelles.

        Vous pouvez également y rappeler le lien avec votre rapport Word ou d'autres documents.
        """
    )


if __name__ == "__main__":
    main()
