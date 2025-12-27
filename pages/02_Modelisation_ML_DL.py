import streamlit as st


def main():
    st.title("Modélisation : ML & DL")

    st.markdown(
        """
        Cette page présente les différentes approches de modélisation et le
        raisonnement qui conduit au choix final des modèles retenus.
        """
    )

    with st.sidebar.expander("Modélisation ML / DL", expanded=False):
        choix_modele = st.radio(
            label="Sélection de rubrique",
            options=("Approche", "Machine Learning", "Deep Learning"),
            label_visibility="collapsed",
        )

    if choix_modele == "Approche":
        st.subheader("Approche globale ML & DL")
        st.markdown(
            """
            Cette section présente l'**approche globale** de modélisation :

            - comment s'articulent les volets *Machine Learning* et *Deep Learning*,
            - pourquoi ces deux familles de modèles ont été étudiées,
            - les grands critères de comparaison (métriques, contraintes, données, ...),
            - la logique générale qui mène aux PoC DL et au choix final.

            Vous pourrez ici donner une vue d'ensemble avant de rentrer dans le détail
            des onglets *Machine Learning* et *Deep Learning*.
            """
        )

    elif choix_modele == "Machine Learning":
        st.subheader("Approches Machine Learning")

        onglet_metho_ml, onglet_features, onglet_analyse, onglet_entrainement, onglet_resultats = st.tabs(
            ["Méthodologie", "Features", "Analyse", "Entraînement", "Résultats"]
        )

        with onglet_metho_ml:
            st.markdown(
                """
                Voici la méthodologie suivie : .
                """
            )

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.image(
                    "/workspaces/app/figures/ML/Image1.jpg",
                    caption="Schéma de la méthodologie Machine Learning",
                    width=600,
                )

        with onglet_features:
            st.markdown(
                """
                Dans cet onglet, vous pouvez détailler vos **features** :

                - type de représentations utilisées (embeddings, statistiques, etc.),
                - transformations appliquées (normalisation, standardisation, encodage, ...),
                - éventuelle sélection / réduction de dimension.
                """
            )

        with onglet_analyse:
            st.markdown(
                """
                Ici, vous pouvez présenter l'**analyse** des modèles ML :

                - comparaison des algorithmes (SVM, Random Forest, XGBoost, ...),
                - analyse d'importance de variables,
                - comportements particuliers observés (overfitting, underfitting, ...).
                """
            )

        with onglet_entrainement:
            st.markdown("### Entraînement des modèles ML")

            st.markdown(
                """
                <style>
                .tile {
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 0.75rem;
                    background-color: #f5f5f5;
                    border: 1px solid #e0e0e0;
                }
                .tile-title {
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                }
                .tile-icon {
                    margin-right: 0.25rem;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    <div class="tile">
                        <div class="tile-title"><span class="tile-icon">⚙️</span>SVM</div>
                        <ul>
                            <li>Pipeline d'entraînement SVM</li>
                            <li>Hyperparamètres principaux</li>
                            <li>Temps de calcul, stabilité, etc.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
                    <div class="tile">
                        <div class="tile-title"><span class="tile-icon">🚀</span>XGBoost</div>
                        <ul>
                            <li>Procédure d'entraînement XGBoost</li>
                            <li>Grilles d'hyperparamètres testées</li>
                            <li>Points forts / limites observés</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    """
                    <div class="tile">
                        <div class="tile-title"><span class="tile-icon">🌲</span>ExtraTrees</div>
                        <ul>
                            <li>Configuration des forêts ExtraTrees</li>
                            <li>Comparaison avec les autres modèles</li>
                            <li>Comportements particuliers</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    """
                    <div class="tile">
                        <div class="tile-title"><span class="tile-icon">📏</span>Log reg</div>
                        <ul>
                            <li>Régression logistique de référence</li>
                            <li>Rôle de baseline</li>
                            <li>Comparaison avec les modèles complexes</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with onglet_resultats:
            st.markdown(
                """
                Dans cet onglet, vous pouvez présenter les **résultats** :

                - métriques principales (sans f1_health si vous ne souhaitez pas l'afficher),
                - tableaux comparatifs entre modèles,
                - visualisations (courbes, matrices de confusion, ...),
                - synthèse des modèles retenus côté Machine Learning.
                """
            )
    else:
        st.subheader("Approches Deep Learning")

        onglet_metho_dl, onglet_archi_dl, onglet_criteres_dl = st.tabs(
            ["Méthodologie", "Architectures", "Critères de sélection"]
        )

        with onglet_metho_dl:
            st.markdown(
                """
                Cette section décrit la **méthodologie générale côté Deep Learning** :

                - choix des familles d'architectures (mono-tâche, multi-tâche, têtes multiples, ...),
                - protocole d'entraînement (finetuning ou non, scheduler, callbacks, ...),
                - stratégie de comparaison entre architectures (stabilité, généralisation, coût, ...),
                - articulation avec les PoC (DL PoC 1, DL PoC 2, ...).
                """
            )

            col_left_dl, col_center_dl, col_right_dl = st.columns([1, 2, 1])
            with col_center_dl:
                st.image(
                    "/workspaces/app/figures/DL/Image2.png",
                    caption="Schéma de la méthodologie Deep Learning",
                    width=600,
                )

        with onglet_archi_dl:
            st.markdown(
                """
                Dans cet onglet, vous pouvez détailler les **architectures Deep Learning** :

                - architectures testées (mono-tâche, multi-tâche, différentes têtes, ...),
                - variantes explorées (backbones, tailles de modèles, etc.),
                - stratégie d'entraînement (finetuning, pas de finetuning, scheduler, ...).
                """
            )

        with onglet_criteres_dl:
            st.markdown(
                """
                Ici, vous pouvez expliciter les **critères de sélection** des architectures DL :

                - stabilité des résultats,
                - capacité de généralisation,
                - coût / complexité du modèle,
                - contraintes de temps de calcul / mémoire,
                - autres critères spécifiques à votre cas d'usage.
                """
            )


if __name__ == "__main__":
    main()
