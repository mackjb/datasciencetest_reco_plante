import streamlit as st


def main():
    st.title("DL PoC 1")

    st.markdown(
        """
        Cette page détaillera le **Cas X** en Deep Learning, par exemple :

        - l'architecture utilisée (backbone, têtes de sortie, etc.),
        - le protocole d'entraînement (hyperparamètres, scheduler, etc.),
        - les principaux résultats (courbes de loss, métriques, matrices de confusion, ...),
        - une interprétation des résultats.

        Vous pourrez intégrer ici des figures générées par vos scripts (PNG, etc.)
        et des commentaires issus de vos analyses.
        """
    )


if __name__ == "__main__":
    main()
