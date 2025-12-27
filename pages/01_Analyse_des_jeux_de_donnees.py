import streamlit as st


def main():
    st.title("Analyse des jeux de données")

    st.markdown(
        """
        Cette page présentera :

        - la description du jeu de données,
        - des visualisations exploratoires (répartition des classes, exemples d'images, etc.),
        - les étapes de prétraitement (nettoyage, splits train/val/test, augmentations, etc.).

        Vous pourrez ici intégrer vos graphiques, tableaux et explications à partir de vos scripts existants
        ou de vos rapports (HTML, Word).
        """
    )


if __name__ == "__main__":
    main()
