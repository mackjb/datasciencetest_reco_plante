import streamlit as st

def sidebar_choice():
    st.title("Conclusion & Perspectives")
    
    st.markdown("**Modèles très performants sur PlantVillage, mais généralisation limitée par les biais du dataset.**")

    st.divider()
    col_bilan, col_acquis = st.columns(2)
    with col_bilan:
        st.subheader("Bilan")
        st.write("Résumé global des résultats obtenus et des enseignements tirés du projet.")

    with col_acquis:
        st.subheader("Acquis")
        st.success("Cycle complet (EDA → features → ML → DL → Grad-CAM)")
        st.success("SVM RBF meilleur en ML classique pour 14 espèces")
        st.success("DL : 9 architectures explorées, 6 retenues, 2 recommandées")
        st.success("Grad-CAM utile pour détecter “performance réelle” vs corrélations")

    st.divider()

    tab_conc, tab_lim, tab_persp = st.tabs(["Conclusions", "Limites", "Perspectives"])

    # --- Conclusions ---

    with tab_conc:
        c1, c2 = st.columns(2)

        with c1:
            st.header("Résultats Clés")
            st.success("**Performance** : Le Deep Learning (Archi 9) atteint un F1-score moyen de **99.55%**.")
            st.info("**ML Classique** : Le SVM-RBF reste une excellente baseline avec **93.7%** d'accuracy sur l'espèce.")
            st.warning("**Limites** : Biais possible sur le fond (studio) et corrélation espèce-maladie propre au dataset.")

        with c2:
            st.header("Impact Métier")
            st.markdown("""
            *   **Gain de temps** : Diagnostic instantané vs expertise humaine coûteuse.
            *   **Scalabilité** : Déploiement possible sur le cloud ou en edge computing.
            *   **Fiabilité** : Standardisation du diagnostic, réduisant l'erreur humaine subjective.
            """)

    # --- Limites ---

    with tab_lim:
        st.header("Limites")
 
        # Injection du CSS des cartes pour la page Conclusion
        st.markdown('''
        <style>
        /* Cards */
        .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        background: rgba(255,255,255,0.03);
        margin: 0.25rem 0 0.9rem 0;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
        }
        [data-theme="light"] .card {
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(0,0,0,0.02);
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        }
        .card__title {
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
        }
        .card__body {
        color: rgba(255,255,255,0.82);
        font-size: 0.95rem;
        line-height: 1.35;
        }
        [data-theme="light"] .card__body { color: rgba(0,0,0,0.74); }
        .card--success { border-color: rgba(46, 204, 113, 0.35); }
        .card--warning { border-color: rgba(241, 196, 15, 0.40); }
        .card--info    { border-color: rgba(52, 152, 219, 0.40); }
        </style>
        ''', unsafe_allow_html=True)

        col_lim1, col_lim2, col_lim3, col_lim4 = st.columns(4)

        with col_lim1:
            st.markdown('''
            <div class="card card--success" style="background-color:#FFE4C4;">
            <div class="card__title">Validation terrain absente</div>
            <div class="card__body" style="color:#0131B4;">Test en conditions agricoles réelles, Validation par experts botanistes, Pilote terrain avec agriculteurs, Feedback utilisateurs finaux non réalisés.<br/> Conséquence, les performances présentées constituent une borne supérieure en conditions contrôlées.</div>
            </div>
            ''', unsafe_allow_html=True)

        with col_lim2:
            st.markdown('''
        <div class="card card--success" style="background-color:#FEC3AC;">
            <div class="card__title">Biais PlanVillage</div>
            <div class="card__body" style="color:#0131B4;">Fond uniforme, éclairage contrôlé, feuilles isolées, angles standardisés.<br/>Conséquence, prévoir une chute de performance -5% à -15% en conditions réelles <br/><br/><br/></div>
            </div>
            ''', unsafe_allow_html=True)

        with col_lim3:
            st.markdown('''
            <div class="card card--success" style="background-color:#FFE4C4;">
            <div class="card__title">Validation statistique limitée</div>
            <div class="card__body" style="color:#0131B4;">5 runs sur l’architecture 7 uniquement pour des problèmes de performances, les résultats montrent un modèle robuste.<br/>Pour les autres architectures : 1 seul run par architecture. <br/> Pas de test statistique (Test de significativité, p_value &lt; 0,05).<br/>Conséquence : les résultats exacts peuvent varier (±0.1-0.2%).</div>
            </div>
            ''', unsafe_allow_html=True)

        with col_lim4:
            st.markdown('''
            <div class="card card--success" style="background-color:#FEC3AC;">
            <div class="card__title">Classes difficiles</div>
            <div class="card__body" style="color:#0131B4;">Faux négatifs : Maladie rare non détectée.<br/> Conséquence : propagation de la maladie. <br/>Faux positifs : Maladie rare erronée. <br/> Conséquence Sur-traitement inutile (coût + écologie)<br/><br/></div>
            </div>
            ''', unsafe_allow_html=True)

    # --- Perspectives & Améliorations ---
    with tab_persp:
        st.header("Perspectives & Améliorations")

        col_p1, col_p2, = st.columns(2)

        with col_p1:
            st.subheader(" ")       
            st.image(
                "Streamlit/assets/perspectives.png",
                width=700,
            )

        with col_p2:
            st.subheader("Datasets")
            st.write("Diversifier, casser corrélation espèce–maladie, dataset conditions 'Wild'.")

            st.subheader("Modèles Pré-entrainés")
            st.write("Effectuer une sélection plus poussée des modèles pré-entraînés: (ex. ResNet, ConvNeXt, Swin Transformer…) pour identifier celui qui offre le meilleur compromis entre précision, généralisation et ressource.")
            st.write("Expérimenter avec des architectures alternatives :**Vision Transformer (ViT)** pour capter des dépendances plus fines.")

            st.subheader("Architecture")
            st.write("Ajuster les têtes de classification repenser la décomposition des tâches, modifier le nombre ou le type de têtes, ajuster les pertes associées aux têtes) afin que le modèle se concentre sur les caractéristiques pertinentes, indépendamment des biais du dataset.")

            st.subheader("Entrainement")
            st.write("Valider la robustesse des architectures : Effectuer plusieurs runs multiples avec des statistiques (moyenne, écart-type) pour vérifier la stabilité et la reproductibilité des performances des architectures retenues.")


