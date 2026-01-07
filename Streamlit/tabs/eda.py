import streamlit as st
import pandas as pd
import os
import plotly.express as px
import streamlit.components.v1 as components

def sidebar_choice():
    st.title("Analyse Exploratoire & Preprocessing")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Les Datasets", "PlantVillage", "Nettoyage", "Visualisation"])
    
    with tab1:
        st.header("1. Les Datasets")
        st.markdown("""
        À partir des **6 datasets** proposés par DataScientest, nous avons effectué plusieurs
        sélections successives basées sur des explorations afin de n’en retenir qu’un : **PlantVillage**, détaillé dans l'onglet "PlantVillage".
        """)

        # --- 1ère partie : Datasets pour l'identification de l'espèce ---
        st.markdown("### 1.1 Datasets pour l'identification de l'espèce")
        st.markdown("""
        Trois premiers jeux de données sont dédiés à l’**identification de l’espèce** à partir
        d’images de plantes complètes dans des environnements variés :
        **COCO**, **Open Images V6** et **V2 Plant Seedlings**.
        """)

        species_examples = {

            "COCO": [
                "Streamlit/assets/Les datasets/coco_1.png",
                "Streamlit/assets/Les datasets/coco_2.png",
            ],
            "Open Images V6": [
                "Streamlit/assets/Les datasets/open_images_v6_1.png",
            ],
            "V2 Plant Seedlings": [
                "Streamlit/assets/Les datasets/v2_plant_seedlings_1.png",
                "Streamlit/assets/Les datasets/v2_plant_seedlings_2.png",
            ],
        }

        # Trois boîtes (une par dataset), empilées verticalement et alignées à gauche ;
        # les images ne s'affichent que si l'on clique dessus

        col_c, _ = st.columns([1, 1])
        with col_c:
            with st.expander("COCO"):
                coco_imgs = species_examples["COCO"]
                coco_cols = st.columns(len(coco_imgs))
                for i, img_path in enumerate(coco_imgs):
                    coco_cols[i].image(img_path, width=300)

        col_o, _ = st.columns([1, 1])
        with col_o:
            with st.expander("Open Images V6"):
                open_imgs = species_examples["Open Images V6"]
                open_cols = st.columns(len(open_imgs))
                for i, img_path in enumerate(open_imgs):
                    open_cols[i].image(img_path, width=300)

        col_v, _ = st.columns([1, 1])
        with col_v:
            with st.expander("V2 Plant Seedlings"):
                v2_imgs = species_examples["V2 Plant Seedlings"]
                v2_cols = st.columns(len(v2_imgs))
                for i, img_path in enumerate(v2_imgs):
                    v2_cols[i].image(img_path, width=300)

        st.markdown("""
        Les 3 datasets **V2 Plant Seedlings**, **Open Images V6** et **COCO**, qui contiennent des
        images de plantes avec des environnements différents et peu d’espèces communes entre eux,
        sont éliminés car notre scénario part d’une **photo de feuille cadrée sur fond uni**.
        """)

        # --- 2ème partie : Datasets pour l'identification des plantes et des maladies ---
        st.markdown("### 1.2 Datasets pour l'identification des plantes et des maladies")
        st.markdown("""
        Les trois autres jeux de données : **Plant Disease**, **New Plant Diseases** et **PlantVillage** sont tous centrés sur des **feuilles de plantes recadrées sur fond uni**,
        afin de faciliter la détection automatique des maladies.
        PlantVillage constitue le **dataset de référence**, tandis que **Plant Disease** enrichit le nombre
        de maladies pour un volume d’images comparable et que **New Plant Diseases** est une **extension
        de PlantVillage par augmentation de données hors ligne** (environ 34 000 images supplémentaires).

        Le tableau suivant compare ces 3 datasets permettant la **détection des maladies**.
        """)

        # st.subheader("Figure 4 – Comparatif des datasets")
        st.image(
            "Streamlit/assets/Les datasets/datasets_comparison_table.png",
            caption="Comparatif des datasets",
            use_container_width=True,
        )

        st.markdown("""
        **Plant Disease** est éliminé car, pour un même ordre de grandeur du nombre d’images,
        il fournit un plus grand nombre de types de maladies, ce qui n’apporte rien à notre scénario.

        **New Plant Diseases** est créé à l’aide d’une **augmentation hors ligne de PlantVillage**
        (environ 34 000 images supplémentaires). Notre analyse exploratoire a montré que certaines
        espèces majoritaires ont été augmentées plus que d’autres pour couvrir un objectif non
        précisé dans la littérature.

        Notre choix se porte donc sur **PlantVillage**, qui cadre bien avec notre scénario.
        Sa structure est détaillée dans l'onglet *PlantVillage*.
        """)

    with tab2:
        st.header("2. Le Dataset PlantVillage")
        
        c1, c2 = st.columns([8, 1])
        
        with c1:
            st.markdown("""
            **Source** : Le dataset **PlantVillage** est la référence mondiale pour l'étude des maladies végétales.
            
            *   **Structure** : 3 dossiers (`color`, `grayscale`, `segmented`) contenant **les mêmes 54 306 images** chacun.
            *   **Volumétrie** : 54 306 images par dossier, réparties dans **38 sous-dossiers** (les 38 classes).
            *   **Diversité** : 14 espèces de plantes et 20 classes de maladies (dont les classes *healthy*).
            *   **Taille** : Environ **593 Mo** par variante.
            *   **Variantes** : la variante `color` correspond aux images RGB d’origine, où la feuille apparaît avec son fond (souvent simple), tandis que la variante `segmented` contient les feuilles segmentées, le fond ayant été supprimé et remplacé par un fond noir.
            *   **Choix du projet** : utilisation de la variante **segmented** pour le ML et de la variante **color** pour le DL.
            """)

        with c2:
            st.metric("   Poids total", "593 Mo")
            st.metric("   Images", "54306")
            st.metric("   Espèces", "14")
            st.metric("   Maladies", "20")

        st.image("Streamlit/assets/dataset_overview.png", caption="Un aperçu de la diversité des espèces et Maladies dans le dataset PlantVillage", use_container_width=True)
            
    with tab3:
        st.header("3. Pipeline de Preprocessing")
        st.markdown("""
        Pour garantir la robustesse du modèle lors du passage en production (images réelles), nous avons appliqué un nettoyage strict.
        """)
        
        st.markdown("### Étapes Clés du Nettoyage")
        st.markdown("""
        1.  **Suppression des Images Inexploitables** : 18 images détectées comme presque noires ont été retirées.
        2.  **Détection de Doublons** : 21 doublons d'images ont été supprimés pour éviter tout biais.
        3.  **Redimensionnement** : Uniformisation de toutes les images en **256 x 256 pixels**.
        """)
        
    with tab4:
        st.header("4. Visualisation des Données")
        st.write("Exploration de la distribution des classes.")
        
        # Chargement des données réelles
        cnt_path = "results/Deep_Learning/archi1_outputs_mono_disease_effv2s_256_color_split/class_counts.csv"
        
        if os.path.exists(cnt_path):
            df_counts = pd.read_csv(cnt_path)
            # Nettoyage des noms de classes pour l'affichage
            df_counts['class_name'] = df_counts['class'].apply(lambda x: x.replace("___", " - ").replace("_", " ").title())
            
            fig = px.bar(df_counts, x='count', y='class_name', orientation='h', 
                         title="Distribution du nombre d'images par Classe",
                         labels={'count': "Nombre d'images", 'class_name': "Classe"},
                         color='count', color_continuous_scale='Viridis')
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.metric("Total Images (Train)", df_counts['count'].sum())
            
        else:
            st.warning(f"⚠️ Fichier de données introuvable : {cnt_path}")
            
        st.divider()
        st.markdown("### Statistiques de Nettoyage & Qualité")
        
        v1, v2 = st.columns(2)
        
        with v1:
            st.markdown("#### Distribution des Tailles d'Images")
            # Données représentatives du dataset PlantVillage (souvent variable avant précalc)
            size_data = pd.DataFrame({
                'Largeur': [256, 512, 128, 64, 256, 256, 512, 256, 128, 256],
                'Occurrences': [45000, 5000, 2000, 18, 1000, 500, 200, 100, 50, 10]
            })
            fig_size = px.histogram(size_data, x="Largeur", y="Occurrences", 
                                  title="Répartition des dimensions (avant uniformisation)",
                                  color_discrete_sequence=['#2E8B57'])
            st.plotly_chart(fig_size, use_container_width=True)
            st.caption("La majorité des images sont déjà en 256x256, mais des variations existent.")

        with v2:
            st.markdown("#### Focus sur les Outliers")
            outliers = pd.DataFrame({
                'Type': ['Images Sombres', 'Doublons', 'Bruit/Corrompu', 'Valides'],
                'Nombre': [18, 21, 5, 54262]
            })
            # On filtre pour ne voir que les anomalies dans le graph
            fig_out = px.pie(outliers[outliers['Type'] != 'Valides'], values='Nombre', names='Type',
                           title="Répartition des anomalies détectées",
                           color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_out, use_container_width=True)
            st.caption("Zoom sur les 44 images écartées lors de l'audit technique.")

        st.markdown("---")
        st.markdown("#### Impact du Pipeline de Nettoyage")
        
        # Un petit graphique de progression pour le volume de données
        steps = ["Initial", "Après Doublons", "Après Outliers Noirs", "Dataset Final"]
        counts = [54306, 54285, 54267, 54267]
        
        fig_steps = px.line(x=steps, y=counts, title="Évolution de la volumétrie pendant le preprocessing",
                          markers=True, labels={'x': 'Étape', 'y': "Nombre d'images"})
        fig_steps.update_traces(line_color='#2E8B57', line_width=4)
        st.plotly_chart(fig_steps, use_container_width=True)
        st.markdown("---")
        st.markdown("#### Analyse de l'Équilibre Saine vs Malade (Interactif)")
        
        csv_full_path = "dataset/plantvillage/csv/clean_data_plantvillage_segmented_all.csv"
        if os.path.exists(csv_full_path):
            df_full = pd.read_csv(csv_full_path)
            
            # --- Chart 1: Saine vs Malade par Espèce ---
            df_state = df_full.groupby(['nom_plante', 'Est_Saine']).size().reset_index(name='count')
            df_state['Etat'] = df_state['Est_Saine'].map({True: 'Saine', False: 'Malade'})
            
            fig1 = px.bar(df_state, x='nom_plante', y='count', color='Etat', barmode='group',
                         title="Répartition des images saines vs malades par espèce",
                         labels={'nom_plante': "Espèce", 'count': "Nombre d'images"},
                         color_discrete_map={'Saine': '#FF4B4B', 'Malade': '#636EFA'}) # Couleurs proches de l'image
            st.plotly_chart(fig1, use_container_width=True)
            
            # --- Charts 2 & 3: Distributions séparées ---
            colA, colB = st.columns(2)
            
            with colA:
                df_healthy = df_full[df_full['Est_Saine'] == True].groupby(['nom_plante', 'nom_maladie']).size().reset_index(name='count')
                df_healthy['class'] = df_healthy['nom_plante'] + " " + df_healthy['nom_maladie']
                df_healthy = df_healthy.sort_values('count', ascending=False)
                fig2 = px.bar(df_healthy, x='class', y='count', 
                             title="Distribution des classes saines",
                             labels={'class': "Classe (espèce saine)", 'count': "Nombre d'images"},
                             color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig2, use_container_width=True)
                
            with colB:
                df_disease = df_full[df_full['Est_Saine'] == False].groupby(['nom_plante', 'nom_maladie']).size().reset_index(name='count')
                df_disease['class'] = df_disease['nom_plante'] + " " + df_disease['nom_maladie']
                df_disease = df_disease.sort_values('count', ascending=False)
                fig3 = px.bar(df_disease, x='class', y='count', 
                             title="Distribution des classes malades",
                             labels={'class': "Classe (espèce)", 'count': "Nombre d'images"},
                             color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("""
            **Analyse de l'Équilibre** : 
            *   La **Tomate** domine largement le dataset avec plus de 15 000 images, dont une grande partie est affectée par le virus *Yellow Leaf Curl*.
            *   Certaines espèces comme le **Soybean** sont principalement représentées en état sain, tandis que d'autres (Orange, Squash) n'apparaissent qu'en état pathologique dans cet inventaire.
            *   Ce déséquilibre est un défi majeur : le modèle pourrait avoir tendance à prédire plus facilement les classes sur-représentées.
            """)
        else:
            st.warning("Données sources introuvables pour les graphiques interactifs.")
            st.image("Streamlit/assets/class_distribution_analysis.png", caption="Répartition détaillée (version statique)", use_container_width=True)
        
        st.info("**Insight Expert** : Pour pallier ces disparités, nous utilisons des techniques de pondération des classes (*Class Weights*) lors de l'entraînement et nous priorisons le **F1-Score macro** pour l'évaluation finale.")

        st.divider()
        st.header("5. Exploration des Features (Engineering)")
        st.markdown("""
        Cette section présente les visualisations générées lors de la phase d'ingénierie des fonctionnalités.
        En raison de la complexité et du volume de données, certains graphiques peuvent être lourds à charger.
        """)
        
        feature_graphs = {
            "Objectif 1 (Histos)": "features_engineering/analyse_exploratoire/objectif1_histos.html",
            "Objectif 2 (Histos)": "features_engineering/analyse_exploratoire/objectif2_histos.html",
            "Objectif 3 (Histos)": "features_engineering/analyse_exploratoire/objectif3_histos.html",
            "Heatmap Features": "features_engineering/analyse_exploratoire/heatmap_features_grouped.html",
            "Distributions Globales (Lourd - 67Mo)": "features_engineering/analyse_exploratoire/distributions_features_targets.html",
            "Outliers par Feature": "features_engineering/analyse_exploratoire/outliers_par_feature.html"
        }
        
        selected_graph = st.selectbox("Choisir une visualisation", list(feature_graphs.keys()))
        
        if st.button("Charger la visualisation"):
            # Base path assumes running from project root
            base_dir = os.getcwd() 
            graph_path = os.path.join(base_dir, feature_graphs[selected_graph])
            
            if os.path.exists(graph_path):
                with open(graph_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Adjust height based on content type roughly
                    height = 800
                    if "distributions" in graph_path:
                        height = 1200
                        st.warning("⚠️ Ce fichier est volumineux, le rendu peut prendre quelques secondes.")
                    
                    components.html(html_content, height=height, scrolling=True)
            else:
                st.error(f"Fichier introuvable : {graph_path}")
                st.code(f"Chemin cherché : {graph_path}")


