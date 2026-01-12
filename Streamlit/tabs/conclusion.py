import streamlit as st
from utils import render_mermaid

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
            "Streamlit/assets/produit.png",
            width=900,
        )

    with col_ret:
        st.markdown("<h4>Ce qu'on a retenu</h4>", unsafe_allow_html=True)
        st.image(
            "Streamlit/assets/retenu.png",
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

        st.divider()

        # --- Limites (Mindmap) ---
        st.subheader("Limites")
        
        mindmap_limites = {
            "id": "root",
            "text": "Limites",
            "children": [
                {
                    "id": "validation_terrain",
                    "text": "Validation terrain absente",
                    "children": [
                        {"id": "terrain_details", "text": "Test en conditions agricoles réelles, validation par experts, pilote terrain non réalisés."},
                        {"id": "terrain_cq", "text": "Conséquence : performances = borne supérieure en conditions contrôlées."}
                    ]
                },
                {
                    "id": "validation_stat",
                    "text": "Validation statistique limitée",
                    "children": [
                        {"id": "stat_details", "text": "5 runs sur Archi 7, 1 seul run sur les autres."},
                        {"id": "stat_cq", "text": "Conséquence : variation possible des résultats (±0.1–0.2 %)."}
                    ]
                },
                {
                    "id": "biais_dataset",
                    "text": "Biais du dataset PlantVillage",
                    "children": [
                        {"id": "biais_details", "text": "Fond uniforme, éclairage contrôlé, feuilles isolées."},
                        {"id": "biais_cq", "text": "Conséquence : chute estimée de −5% à −15% en conditions réelles."}
                    ]
                },
                {
                    "id": "backbone",
                    "text": "Un seul backbone",
                    "children": [
                        {"id": "EfficientNetV2S", "text": "Voir pour permettre de tester plusieurs backbones"}
                    ]
                }
            ]
        }
        render_mermaid(mindmap_limites, height=600)

        st.divider()

        # --- Acquis & Perspectives ---
        # Injection CSS pour les cards (déplacé ici ou global)
        st.markdown('''
        <style>
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

        # --- Acquis & Perspectives ---
        col_persp, col_acquis = st.columns(2)

        with col_acquis:
            with st.expander("Acquis", expanded=True):
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
            with st.expander("Perspectives & Améliorations", expanded=True):
                # Image centrée en haut via colonnes
                col_img_left, col_img_center, col_img_right = st.columns([1, 2, 1])
                with col_img_center:
                    st.image(
                        "Streamlit/assets/perspectives.png",
                        width=500,
                    )

                # Textes en dessous
                st.write("**Datasets**")
                st.write("Diversifier, casser corrélation espèce–maladie, dataset conditions 'Wild'.")

                st.write("**Modèles Pré-entrainés**")
                st.write("Effectuer une sélection plus poussée (ex. ResNet, ConvNeXt, Swin Transformer…). Expérimenter avec **Vision Transformer (ViT)**.")

                st.write("**Architecture**")
                st.write("Ajuster les têtes de classification, repenser la décomposition des tâches.")

                st.write("**Entrainement**")
                st.write("Valider la robustesse : Runs multiples avec statistiques pour vérifier la stabilité.")

    st.divider()

    # --- Slide Questions ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            text-align: center; 
            background-color: #e3f2fd; 
            padding: 40px; 
            border-radius: 15px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
        ">
            <h1 style="color: #1565c0; margin-bottom: 20px;">Questions / Réponses</h1>
            <div style="font-size: 80px;">❓</div>
            <p style="font-size: 1.5rem; color: #5c6bc0; margin-top: 20px; font-weight: bold;">
                Merci de votre attention !
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )