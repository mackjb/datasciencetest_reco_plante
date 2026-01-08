import streamlit as st
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

def sidebar_choice():
    st.title("Conclusion & Perspectives")

    st.markdown(
        """
        <style>
        /* Couleur du titre dans le header de tous les expanders */
        div[data-testid="stExpander"] summary p {
            color: #0131B4;
            font-weight: 700;
        }
        /* Fond du header de tous les expanders (ouvert ou fermé) */
        div[data-testid="stExpander"] summary {
            background-color: #ffe2d1 !important;
            border-radius: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Bilan")

    col_prod, col_ret = st.columns(2)

    with col_prod:
        st.markdown("<h4>Ce qu'on a produit</h4>", unsafe_allow_html=True)
        st.image(
            os.path.join(ASSETS_DIR, "produit.png"),
            width=900,
        )

    with col_ret:
        st.markdown("<h4>Ce qu'on a retenu</h4>", unsafe_allow_html=True)
        st.image(
            os.path.join(ASSETS_DIR, "retenu.png"),
            width=900,
        )

    st.divider()

    # --- Bloc Conclusions + Limites & Perspectives ---
    with st.container():
        st.subheader("Conclusions")
        st.markdown(
            """
            - Niveau atteint sur PlantVillage-like :Les meilleures architectures convergent vers ~99.5% de F1 : en contexte contrôlé, le diagnostic automatisé est très performant.
            - Le meilleur modèle dépend du contexte (scénario) Archi3 = meilleur compromis pour mobile/edge ; Archi9 = choix robuste pour usage pro.
            - On a optimisé une décision de déploiement avec un arbitrage explicite : précision ↔ complexité ↔ latence ↔ maintenance.
            - Les métriques seules ne suffisent pas : on a regardé “où le modèle regarde” Les visualisations (Grad-CAM) permettent de juger si le modèle s’appuie sur les lésions ou sur des corrélations.
            - Positionnement clair : “assistant IA en usage contrôlé” Recommandé si conditions proches PlantVillage ; à encadrer et monitorer dès qu’on s’éloigne du cadre.
            """
        )

        # --- Limites & Perspectives côte à côte ---
        col_lim, col_persp = st.columns(2)

        with col_lim:
            st.markdown('<div id="expander-limites">', unsafe_allow_html=True)
            with st.expander("Limites"):

                # Injection du CSS des cartes pour la page Conclusion
                st.markdown('''
                <style>
                    /* Cards */
                    .card {
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 18px;
                    padding: 16px 16px 14px 16px;
                    background: #cddafd;
                    margin: 0.25rem 0 0.9rem 0;
                    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
                    }
                    [data-theme="light"] .card {
                    border: 1px solid rgba(0,0,0,0.08);
                    background: #8eecf5;
                    box-shadow: 0 8px 22px rgba(0,0,0,0.06);
                    }
                    .card__title {
                    font-weight: 800;
                    font-size: 1.05rem;
                    margin-bottom: 0.35rem;
                    }
                    .card__body {
                    color: rgba(0,0,0,0.82);
                    font-size: 0.95rem;
                    line-height: 1.35;
                    }
                    .card--success { border-color: rgba(46, 204, 113, 0.35); }
                    .card--warning { border-color: rgba(241, 196, 15, 0.40); }
                    .card--info    { border-color: rgba(52, 152, 219, 0.40); }
                    </style>
                ''', unsafe_allow_html=True)

                st.markdown(
                    """
                    **Validation terrain absente**  
                    Test en conditions agricoles réelles, validation par experts botanistes, pilote terrain avec agriculteurs, feedback utilisateurs finaux non réalisés.  
                    → Conséquence : les performances présentées constituent une borne supérieure en conditions contrôlées.

                    ---

                    **Validation statistique limitée**  
                    5 runs sur l’architecture 7 uniquement pour des problèmes de performances, les résultats montrent un modèle robuste.  
                    Pour les autres architectures : 1 seul run par architecture, seed aléatoire unique, pas de test statistique (test de significativité, p_value < 0,05).  
                    → Conséquence : les résultats exacts peuvent varier (±0.1–0.2 %).

                    ---

                    **Biais du dataset PlantVillage**  
                    Fond uniforme, éclairage contrôlé, feuilles isolées, angles standardisés.  
                    → Conséquence : prévoir une chute de performance d’environ −5 % à −15 % en conditions réelles.

                    ---

                    **Classes maladies difficiles**  
                    Faux négatifs : maladie rare non détectée → propagation de la maladie.  
                    Faux positifs : maladie rare erronée → sur-traitement inutile (coût + écologie).
                    """
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # Acquis dans la même colonne que Limites (version cards)
            with st.expander("Acquis"):
                st.markdown(
                    """
                    <div class="card card--success">
                        <div class="card__title">Pipeline complet</div>
                        <div class="card__body">
                            EDA → préparation données → entraînement → évaluation → interprétabilité → décision déploiement.
                        </div>
                    </div>

                    <div class="card card--info">
                        <div class="card__title">Consolidation de nos compétences</div>
                        <div class="card__body">
                            Traitement des données, Machine Learning, Deep Learning, choix de métriques, interprétabilité.
                        </div>
                    </div>

                    <div class="card card--warning">
                        <div class="card__title">Expérimentation & outillage</div>
                        <div class="card__body">
                            Gestion des environnements (Colab, Codespaces, GPU), reproductibilité (seeds, runs), premiers réflexes MLOps (tracking type MLflow, versioning).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_persp:
            st.markdown('<div id="expander-persp">', unsafe_allow_html=True)
            with st.expander("Perspectives & Améliorations"):

                # Image centrée en haut via colonnes
                col_img_left, col_img_center, col_img_right = st.columns([1, 2, 1])
                with col_img_center:
                    st.image(
                        os.path.join(ASSETS_DIR, "perspectives.png"),
                        width=500,
                    )

                # Textes en dessous
                st.write("**Datasets**")
                st.write("Diversifier, casser corrélation espèce–maladie, dataset conditions 'Wild'.")

                st.write("**Modèles Pré-entrainés**")
                st.write("Effectuer une sélection plus poussée des modèles pré-entraînés: (ex. ResNet, ConvNeXt, Swin Transformer…) pour identifier celui qui offre le meilleur compromis entre précision, généralisation et ressource.")
                st.write("Expérimenter avec des architectures alternatives :**Vision Transformer (ViT)** pour capter des dépendances plus fines.")

                st.write("**Architecture**")
                st.write("Ajuster les têtes de classification repenser la décomposition des tâches, modifier le nombre ou le type de têtes, ajuster les pertes associées aux têtes) afin que le modèle se concentre sur les caractéristiques pertinentes, indépendamment des biais du dataset.")

                st.write("**Entrainement**")
                st.write("Valider la robustesse des architectures : Effectuer plusieurs runs multiples avec des statistiques (moyenne, écart-type) pour vérifier la stabilité et la reproductibilité des performances des architectures retenues.")

            st.markdown("</div>", unsafe_allow_html=True)
