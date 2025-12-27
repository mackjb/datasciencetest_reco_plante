import streamlit as st


def main():
    st.title("Conclusion & Perspectives")

    st.markdown(
        """
        Cette page synthétisera :

        - les principaux résultats obtenus,
        - les enseignements tirés (quels modèles fonctionnent le mieux, dans quelles conditions, ...),
        - les limites de l'étude actuelle,
        - les pistes d'amélioration et perspectives (données supplémentaires, nouvelles architectures, etc.).
        """
    )


if __name__ == "__main__":
    main()
