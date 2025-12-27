import streamlit as st


def main():
    st.title("DL PoC 2")

    st.markdown(
        """
        Cette page détaillera le **Cas Y** en Deep Learning, par exemple :

        - une autre architecture ou variante (multi-tâche, autre backbone, autre stratégie d'entraînement, ...),
        - les résultats associés,
        - la comparaison avec le Cas X,
        - les points forts / limites de cette approche.
        """
    )


if __name__ == "__main__":
    main()
