import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import base64
from utils import render_mermaid


# try/except block removed


ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

def _render_loss_hover(loss_path: str, arch_num: str) -> None:
    if not os.path.exists(loss_path):
        return

    with open(loss_path, "rb") as f:
        loss_data = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .loss-hover-box {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .loss-hover-content {{
            display: none;
            position: absolute;
            top: 120%;
            left: 0;
            background-color: #ffffff;
            padding: 8px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 1000;
        }}
        .loss-hover-box:hover .loss-hover-content {{
            display: block;
        }}
        </style>

        <div class="loss-hover-box">
          <span>Survoler pour voir la courbe de loss - Archi {arch_num}</span>
          <div class="loss-hover-content">
            <img src="data:image/png;base64,{loss_data}" style="max-width:700px;width:100%;height:auto;" />
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_exploration_roadmap():
    # Session state key for the consolidated page
    if 'dl_step' not in st.session_state:
        st.session_state['dl_step'] = 1

    # STEPS: Titles from MIX Version (which are from New Version)
    steps = {
        1: "Exploration Individuelle",
        2: "Démarche Structurée",
        3: "Transfer Learning",
        4: "Architectures",
        5: "Performances",
        6: "Interprétabilité"
    }

    # CSS
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    div.stButton > button[kind="primary"] {
        background-image: linear-gradient(#2e7d32, #1b5e20) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #f0f2f6 !important;
        color: #31333F !important;
        border: 1px solid #d6d6d6 !important;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    # Navigation
    # 6 steps -> 6 buttons + 5 spacers = 11 columns
    cols = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2])
    for idx, (step_id, step_label) in enumerate(steps.items()):
        col_idx = idx * 2
        with cols[col_idx]:
            is_active = (st.session_state['dl_step'] == step_id)
            if st.button(step_label, key=f"dl_flow_btn_{step_id}", type="primary" if is_active else "secondary", width="stretch"):
                st.session_state['dl_step'] = step_id
                st.rerun()
        if idx < len(steps) - 1:
            with cols[col_idx + 1]:
                st.markdown("<h3 style='text-align: center; margin: 0; color: #b0b2b6;'>➜</h3>", unsafe_allow_html=True)

    st.divider()

    current = st.session_state['dl_step']

    # --- STEP 1 ---
    if current == 1:
        st.markdown("""
        Chaque membre de l'équipe a d'abord exploré individuellement 
        un modèle pré-entraîné pour se familiariser avec les techniques de Deep Learning.
        """)
        
        col_img, col_txt = st.columns([1.3, 2])
        with col_img:
            st.image("Streamlit/assets/leviers_DL.png", width="stretch")
        with col_txt:
            st.markdown("### Objectifs & Défis")
            st.markdown("""
            Nous avons chacun travaillé sur des notebooks séparés pour comprendre les impacts de :
            - **Le choix du backbone** (VGG, ResNet, EfficientNet...)
            - **Le Fine-Tuning** : Gel partiel vs total des couches.
            - **L'Augmentation de données** : Impact sur l'overfitting.
            - **Le Déséquilibre des classes** : Utilisation de class_weights.
            """)

        st.markdown(
            """
            <div style="margin: 1rem auto 0 auto; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #ffddd2; border: 1px solid #c8e6c9; width: 75%;">
                <span style="font-weight: 400; color: #0d47a1;">
                    Cette étape était cruciale pour harmoniser nos connaissances avant de définir une architecture commune.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- STEP 2 ---
    elif current == 2:        
        st.markdown("""
        Pour structurer notre approche, nous avons défini **3 cas d'usage** correspondant à différents niveaux de complexité métier.
        """)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div style="background-color: #bde0fe; padding: 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid #d0d4e4;">
                    <p style="margin: 0; font-size: 0.9rem; color: #0d00a4;"><span style="font-weight: 600; color: #0d00a4;">Cas 1 - </span>Identifier l'espèce uniquement</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div style="background-color: #bde0fe; padding: 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid #d0d4e4;">
                    <p style="margin: 0; font-size: 0.9rem; color: #0d00a4;"><span style="font-weight: 600; color: #0d00a4;">Cas 2 - </span>On connait l'espèce → identifier la maladie</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div style="background-color: #bde0fe; padding: 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid #d0d4e4;">
                    <p style="margin: 0; font-size: 0.9rem; color: #0d00a4;"><span style="font-weight: 600; color: #0d00a4;">Cas 3 - </span>Identifier l'espèce et la maladie</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.divider()    

        st.markdown("### Critères de Sélection des Architectures")
        st.table(pd.DataFrame({
            "Catégorie": ["Métier", "Métier", "Technique", "Technique", "Autres"],
            "Critère": ["Précision (Macro-F1)", "Généralisation", "Coût Inférence", "Complexité", "Interprétabilité"],
            "Importance": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐⭐⭐"]
        }))

    # --- STEP 3 ---
    elif current == 3:
        st.subheader("Pourquoi le Transfer Learning ? ")
        # Mindmap Théorie du Transfer Learning
        mindmap_tl = {
            "id": "root",
            "text": "Transfer Learning",
            "children": [
                {
                    "id": "Concept", 
                    "text": "<b>1. Concept & Principes</b>",
                    "collapsed": False,
                    "children": [
                        {"id": "Pretraining", "text": "Exploitation de modèles pré-entraînés sur <i>ImageNet</i> (millions d'images)"},
                        {"id": "Features", "text": "Réutilisation des 'poids' pour détecter des motifs génériques (bords, textures, formes)"},
                        {"id": "Automation", "text": "Extraction automatique des caractéristiques sans intervention manuelle"}
                    ]
                },
                {
                    "id": "Avantages", 
                    "text": "<b>2. Avantages Stratégiques</b>",
                    "collapsed": True,
                    "children": [
                        {"id": "Efficience", "text": "Gain massif en temps d'entraînement et ressources de calcul (GPU/VRAM)"},
                        {"id": "Performance", "text": "Précision élevée même avec un dataset spécialisé plus restreint"},
                        {"id": "Convergence", "text": "Stabilisation plus rapide de l'apprentissage (Loss)"}
                    ]
                },
                {
                    "id": "Mécanique", 
                    "text": "<b>3. Protocole en 2 Phases</b>",
                    "collapsed": True,
                    "children": [
                        {"id": "Phase1", "text": "<b>Phase 1 : Warm-up</b><br/>Backbone gelé, entraînement des nouvelles têtes de classification uniquement"},
                        {"id": "Phase2", "text": "<b>Phase 2 : Fine-tuning</b><br/>Dégel partiel du backbone avec un Learning Rate très réduit pour la spécialisation"}
                    ]
                },
                {
                    "id": "Architectures", 
                    "text": "<b>4. Architectures de référence</b>",
                    "collapsed": True,
                    "children": [
                        {"id": "Petits", "text": "Modèles compacts pour le coût/latence : YoloV8, MobileNet, Petits EfficientNet"},
                        {"id": "Gros", "text": "Gros modèles pour la précision: VGGEfficientNetV2S, ResNet50, DenseNet-121, Vi"}
                    ]
                },
            ]
        }
        render_mermaid(mindmap_tl, height=600)

        st.subheader("Choix du Modèle Pré-entraîné")
        st.markdown("**Comparatif des Modèles Pré-entraînés Explorés :**")
        
        path_carac = os.path.join(ASSETS_DIR, "carac_mod_pre_trained.xlsx")
        if os.path.exists(path_carac):
             df_models = pd.read_excel(path_carac)
             # Stylisation du header via Pandas Styler
             st.table(
                 df_models.set_index("Caractéristique").style.set_table_styles(
                     [{'selector': 'th', 'props': [('background-color', '#e6f2ff'), ('color', '#0d00a4')]}]
                 )
             )
        else:
             st.error("Fichier de données 'carac_mod_pre_trained.xlsx' introuvable.")
        
        st.markdown(
            """
            <div style="margin: 1rem auto 0 auto; width: 66%; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #ffc9b9; border: 1px solid #c8e6c9; color: #0d00a4;">
                <p style="margin: 0 0 0.5rem 0; font-weight: 600;">
                    Backbone pré-entrainé retenu : EfficientNetV2S
                </p>
                <ul style="margin: 0; padding-left: 1.2rem;">
                    <li><b>Performance</b> : Meilleure Accuracy ImageNet (83.9%) parmi les modèles testés.</li>
                    <li><b>Efficience</b> : Excellent ratio performance/paramètres (21.5M).</li>
                    <li><b>Rapidité</b> : Optimisé pour une inférence rapide sur GPU.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

#------- Protocole -------
        st.subheader("Protocole expérimental commun pour les architectures à concevoir")
        
        st.markdown("""
        - Dataset : PlantVillage/color
        - Backbone pré-entraîné : **EfficientNetV2S** (ImageNet)
        - Splits identiques pour tous les modèles
        - Hyperparamètres fixés : learning rate, batch size, augmentation
        - Métriques : Loss, Accuracy, Macro-F1, matrice de confusion
        """)

    # --- STEP 4 ---
    elif current == 4:
        #------- Carte des Architectures -------
        st.subheader("Carte des Architectures")
        st.markdown(
            """
            Nos explorations ont abouti à la conception de **9 architectures distinctes**.
            Le graphique interactif ci-dessous les positionne dans notre *Espace de Conception*, structuré par le degré
             d'isolation des tâches et le niveau de mutualisation du backbone.
            """
        )

        # --- Interactive Scatter Plot for Architectures ---
        # Data preparation
        arch_data = [
            # Dedicated (Top-Left region)
            {"id": 1, "x": 1.5, "y": 5.5, "label": "Archi 1", "group": "Dédié", "desc": "3 modèles indépendants"},
            {"id": 2, "x": 2.5, "y": 4.5, "label": "Archi 2", "group": "Dédié", "desc": "2 modèles (Espèce + Disease_Ext)"},
            {"id": 3, "x": 1.0, "y": 4.0, "label": "Archi 3", "group": "Dédié", "desc": "Modèle unifié (35 classes)"},
            {"id": 4, "x": 3.0, "y": 5.0, "label": "Archi 4", "group": "Dédié", "desc": "Cascade (Espèce -> Maladie)"},
            
            # Shared (Bottom-Right region)
            {"id": 5, "x": 4.5, "y": 2.5, "label": "Archi 5", "group": "Partagé", "desc": "CNN + SVM"},
            {"id": 6, "x": 5.5, "y": 1.5, "label": "Archi 6", "group": "Partagé", "desc": "Multi-tâche unifié"},
            {"id": 7, "x": 5.0, "y": 3.0, "label": "Archi 7", "group": "Partagé", "desc": "Multi-tâche 2 têtes + signal"},
            {"id": 8, "x": 6.0, "y": 2.0, "label": "Archi 8", "group": "Partagé", "desc": "Multi-tâche simplifié"},
            {"id": 9, "x": 4.0, "y": 1.0, "label": "Archi 9", "group": "Partagé", "desc": "Conditionnée (Species+Health->Disease)"},
        ]
        
        df_arch_plot = pd.DataFrame(arch_data)
        
        fig = go.Figure()
        
        # Colors
        color_map = {"Dédié": "#fb5607", "Partagé": "#1565c0"}
        
        # Add traces
        for group in ["Dédié", "Partagé"]:
            subset = df_arch_plot[df_arch_plot["group"] == group]
            fig.add_trace(go.Scatter(
                x=subset["x"], y=subset["y"],
                mode='markers+text',
                text=subset["label"],
                textposition="top center",
                marker=dict(size=18, color=color_map[group], line=dict(width=2, color='white')),
                name=group,
                customdata=subset[["desc", "id"]].values,
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<extra></extra>"
            ))
            
        # Decision Boundary (Oblique)
        # y = x is the separator approx.
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=7, y1=7,
            line=dict(color="gray", width=2, dash="dash"),
        )
        
        # Annotations for regions
        fig.add_annotation(
            x=1.5, y=6.5,
            text="Backbone Dédié à<br>Chaque Objectif",
            showarrow=False,
            font=dict(size=14, color="#fb5607", weight="bold"),
            align="center"
        )
        fig.add_annotation(
            x=5.5, y=0.5,
            text="Backbone Partagé Entre<br>Plusieurs Objectifs",
            showarrow=False,
            font=dict(size=14, color="#1565c0", weight="bold"),
            align="center"
        )
        
        # Annotation for Decision Boundary
        fig.add_annotation(
            x=3.5, y=3.5,
            text="Frontière de Décision",
            textangle=-45,
            showarrow=False,
            font=dict(size=12, color="gray", style="italic")
        )

        fig.update_layout(
            xaxis=dict(range=[0, 7], showgrid=False, zeroline=False, showticklabels=False, title="<b>Niveau de Mutualisation du Backbone (Synergie)</b> →"),
            yaxis=dict(range=[0, 7], showgrid=False, zeroline=False, showticklabels=False, title="<b>Degré d'Isolation des Tâches</b> ↑"),
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(240,242,246,0.5)", # Slight background
            shapes=[
                # Optional: Background colors for regions if desired, kept simple for now
            ]
        )
        
        # Render Plot
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="arch_plot", width="stretch")
        
        selected_point_info = None
        if event and event.get("selection") and event["selection"].get("points"):
            point = event["selection"]["points"][0]
            # customdata should be available if passed to the trace
            if "customdata" in point:
                desc = point["customdata"][0]
                arch_id = point["customdata"][1]
                # Ensure proper ID format
                try:
                    arch_id = int(arch_id)
                except (ValueError, TypeError):
                    pass
                selected_point_info = {"id": arch_id, "desc": desc}
        
        # Handle Selection
        if selected_point_info:
            arch_id = selected_point_info['id']
            arch_desc = selected_point_info['desc']
            
            st.info(f" **Focus : Architecture {arch_id}** — {arch_desc}")
            
            # Display Architecture Image
            img_filename = f"archi_{arch_id}_bk.png"
            img_path = os.path.join(ASSETS_DIR, "architectures", img_filename)
            loss_filename = f"loss_archi_{arch_id}.png"
            loss_path = os.path.join(ASSETS_DIR, "architectures", loss_filename)
            
            # Métriques manuelles (à éditer ici selon besoins)
            metrics_html = ""
            
            # Dictionnaire des performances par architecture
            # Format : { id: {"esp": "VAL_ESP", "mal": "VAL_MAL"} }
            arch_metrics = {
                1: {"esp": "99,87%", "mal": "99,01%"},
                2: {"esp": "99,90%", "mal": "99,22%"},
                3: {"esp": "99,87%", "mal": "99,21%"},
                4: {"esp": "99,88%", "mal": "99,11%"},
                5: {"esp": "99,09%", "mal": "95,45%"},
                6: {"esp": "99,88%", "mal": "98,89%"},
                7: {"esp": "99,85%", "mal": "99,04%"},
                8: {"esp": "99,86%", "mal": "99,08%"},
                9: {"esp": "99,86%", "mal": "99,13%"},
            }

            if arch_id in arch_metrics:
                vals = arch_metrics[arch_id]
                metrics_html = f"""
                <div style="
                    background-color: #90e0ef; 
                    border: 1px solid #e0e0e0; 
                    border-radius: 8px; 
                    padding: 10px; 
                    margin-top: 10px; 
                    text-align: center; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p style="margin: 0; color: #0d00a4; font-size: 0.95rem;"><b>Espèce Macro-F1</b> = {vals['esp']}</p>
                    <p style="margin: 5px 0 0 0; color: #0d00a4; font-size: 0.95rem;"><b>Maladie Macro-F1</b> = {vals['mal']}</p>
                </div>
                """

            if os.path.exists(img_path):
                if os.path.exists(loss_path):
                    # Affichage côte à côte : Architecture + Loss
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(img_path, caption=f"Schéma de l'Architecture {arch_id}", width="stretch")
                    with c2:
                        st.image(loss_path, caption=f"Courbe Loss - Archi {arch_id}", width=300)
                        if metrics_html:
                            st.markdown(metrics_html, unsafe_allow_html=True)
                else:
                    # Centrage et réduction de la taille (50% environ) si pas de loss
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.image(img_path, caption=f"Schéma de l'Architecture {arch_id}", width="stretch")
                        if metrics_html:
                            st.markdown(metrics_html, unsafe_allow_html=True)
            else:
                st.warning(f"Image non trouvée : {img_filename}")
            
            # Tip to scroll down
            st.markdown(f"👇 *Retrouvez les détails complets de l'Archi {arch_id} dans les sections ci-dessous.*")

        st.divider()

        #--------------Les architectures -----------------------
        # Marqueur 1 : début de la zone architectures 1 à 4 (vert)
        st.markdown('<div class="dl-arch-marker-1"></div>', unsafe_allow_html=True)

        # Présentation des architectures 1 à 4 (fond vert)
        st.subheader("Backbone Pré-entraîné Dédié à Chaque Objectif")
        
        arch_info_dedicated = [
            {
                "num": "1",
                "nom": "Trois modèles indépendants",
                "desc": "**Architecture spécialisée** : Trois modèles CNN indépendants, chacun dédié à une seule tâche (species, health, disease). Chaque modèle comprend un backbone pré-entraîné et une tête de classification Dense adaptée au nombre de classes.",
                "workflow": "Chaque modèle s'entraîne en 2 phases sur le même dataset : (1) backbone gelé avec entraînement de la tête uniquement; (2) fine-tuning des dernières couches du backbone pour adapter les features ImageNet aux spécificités du dataset.",
                "avantages": "Simplicité (1 tâche = 1 modèle), absence de conflits entre tâches (pas de compromis dans l'optimisation), performances maximales par tâche (spécialisation totale), interprétabilité facilitée (1 objectif clair par modèle).",
                "limites": "Triplication des ressources (3 backbones à stocker et maintenir), inférences multiples pour cas d'usage complexes, absence de synergie inter-tâches (pas de transfert d'apprentissage entre les 3 têtes), temps d'entraînement cumulé plus long (3 runs).",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_1_bk.png")
            },
            {
                "num": "2",
                "nom": "Deux modèles (species + disease_extended)",
                "desc": "Deux modèles CNN indépendants : l'un pour l'espèce, l'autre pour l'état sanitaire complet. La classe 'healthy' est intégrée comme une maladie spéciale.",
                "workflow": "Deux runs mono-tâche. Le modèle species s'entraîne sur toutes les images (saines + malades). Le modèle disease_extended s'entraîne également sur toutes les images.",
                "avantages": "Simplicité (2 têtes), uniformité (deux softmax multi-classe), diagnostic complet en 2 inférences (species + disease_extended), 'healthy' est un état sanitaire comme les maladies.",
                "limites": "Déséquilibre accru (classe 'healthy' majoritaire), perte de la métrique binaire explicite healthy/diseased, interprétation plus ambiguë des prédictions mixtes (ex: 40% healthy, 35% early_blight).",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_2_bk.png")
            },
            {
                "num": "3",
                "nom": "Modèle unifié (35 classes)",
                "desc": "**Architecture unifiée** : Un modèle CNN pré-entraîné + 1 tête Dense softmax (35 classes). Étiquette combinée : chaque image est étiquetée par un couple 'Espèce__État' (Tomato__healthy, Apple__scab…).",
                "workflow": "Phase 1: backbone gelé, entraînement de la tête uniquement. Phase 2: fine-tuning partiel des dernières couches du backbone. Les labels sont pré-combinés en 35 classes.",
                "avantages": "Un seul modèle, une seule inférence : plus simple à déployer et à utiliser. Synergie entre tâches : l'apprentissage capte directement les co-dépendances espèce↔maladie/santé.",
                "limites": "Moins de spécialisation par tâche. Les classes rares peuvent être sous-apprises. Peu flexible : impossible de gérer des paires inédites (nouvelle espèce/maladie) sans réentraîner les 35 classes. Interprétabilité : plus dur d'isoler l'erreur (vient-elle de l'identification d'espèce ou de maladie ?).",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_3_bk.png")
            },
            {
                "num": "4",
                "nom": "Architecture en cascade",
                "desc": "**Architecture en cascade** : Deux modèles CNN pré-entraînés chaînés. Un classificateur d'espèce extrait un embedding et prédit l'espèce. Un classificateur de maladie global (21 classes, dont 'healthy') reçoit l'image + l'espèce et applique une attention spatiale pour se focaliser sur les zones pertinentes.",
                "workflow": "Phase 1 : backbone gelé, entraînement de la tête. Phase 2 : fine-tuning partiel du backbone. Entraînement du modèle maladie en 2 phases, en lui fournissant l'espèce (True) en entrée pour stabiliser l'apprentissage. Évaluation en CASCADE avec espèce prédite.",
                "avantages": "La prédiction d'espèce guide la maladie, réduisant les confusions entre espèces. L'attention spatiale aide à capter les indices visuels pertinents. Modularité : possibilité d'améliorer séparément espèce ou maladie sans tout réentraîner.",
                "limites": "Une espèce mal prédite dégrade la maladie. Le modèle maladie voit l'espèce (True) à l'entraînement mais la prédite en production. Latence accrue avec passes réseau successives. En cas d'espèce erronée, une maladie impossible peut être proposée.",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_4_bk.png")
            }
        ]
        
        # Affichage des 4 architectures dédiées en grille 2x2
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)

        # Première ligne : architectures 1 et 2
        for col, arch in zip(row1_cols, arch_info_dedicated[:2]):
            with col:
                with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                    left_col, right_col = st.columns(2)

                    # Colonne gauche : schéma
                    with left_col:
                        if os.path.exists(arch['img']):
                            st.image(arch['img'], width="stretch")

                    # Colonne droite : survol pour la courbe de loss
                    with right_col:
                        loss_path = os.path.join(ASSETS_DIR, f"architectures/loss_archi_{arch['num']}.png")
                        if os.path.exists(loss_path):
                            st.caption(f"Courbe d'apprentissage - Archi {arch['num']}")
                            _render_loss_hover(loss_path, arch['num'])
                        
                        st.divider()
                        st.markdown(f"**Workflow** : {arch['workflow']}")
                        st.markdown(f"**Avantages** : {arch['avantages']}")
                        st.markdown(f"**Limites** : {arch['limites']}")

        # Deuxième ligne : architectures 3 et 4
        for col, arch in zip(row2_cols, arch_info_dedicated[2:]):
            with col:
                with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                    left_col, right_col = st.columns(2)

                    # Colonne gauche : schéma
                    with left_col:
                        if os.path.exists(arch['img']):
                            st.image(arch['img'], width="stretch")

                    # Colonne droite : survol pour la courbe de loss
                    with right_col:
                        loss_path = os.path.join(ASSETS_DIR, f"architectures/loss_archi_{arch['num']}.png")
                        if os.path.exists(loss_path):
                            st.caption(f"Courbe d'apprentissage - Archi {arch['num']}")
                            _render_loss_hover(loss_path, arch['num'])
                        
                        st.divider()
                        st.markdown(f"**Workflow** : {arch['workflow']}")
                        st.markdown(f"**Avantages** : {arch['avantages']}")
                        st.markdown(f"**Limites** : {arch['limites']}")
        
        st.divider()

        # Marqueur 2 : début de la zone architectures 5 à 9 (rose/bleu)
        st.markdown('<div class="dl-arch-marker-2"></div>', unsafe_allow_html=True)

        # Architectures 5 à 9 (fond bleu/rose)
        st.subheader("Backbone Pré-entraîné Partagé Entre Plusieurs Objectifs")
        
        arch_info_shared = [
            {
                "num": "5",
                "nom": "CNN + SVM",
                "desc": "**Architecture 'CNN + SVM'** : Un backbone CNN pré-entraîné (gelé) transforme chaque image en vecteur d'embeddings (features). Des classifieurs SVM (espèce, santé, maladie) sont entraînés sur ces embeddings.",
                "workflow": "Sauvegarde des vecteurs + labels. Puis chargement des embeddings, entraînement de trois têtes SVM: Espèce (multi-classe), Santé (binaire: healthy vs diseased), Maladie (soit global multi-classe, soit par espèce).",
                "avantages": "Entraînement très rapide des SVM; itérations légères (on réutilise les embeddings). Simplicité opérationnelle : séparation claire 'features gelées' / 'classifieurs'; facile de remplacer le backbone ou de réentraîner seulement les SVM.",
                "limites": "Les features restent génériques : pas d'adaptation conjointe aux tâches du dataset. Cohérence multi-tâches limitée.",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_5_bk.png")
            },
            {
                "num": "6",
                "nom": "Multi-tâche unifié (3 têtes)",
                "desc": "**Architecture multi-tâche unifiée** : Un seul backbone CNN pré-entraîné partagé produit un embedding commun, puis trois têtes de classification parallèles: Espèce, Santé, Maladie. La tête 'maladie' est optimisée sur les images malades uniquement.",
                "workflow": "Une seule phase 'têtes seules' avec backbone gelé (pertes pondérées par tête). Pas de fine-tuning activé.",
                "avantages": "Les trois tâches se renforcent (l'espèce et la santé aident la maladie). Un seul backbone à entraîner ; une seule inférence pour obtenir espèce, santé, maladie. Contrôle des compromis via pondérations de pertes par tête.",
                "limites": "Conflits d'optimisation : objectifs parfois concurrents ; sensibilité aux pondérations des pertes. Malgré la tête dédiée, les maladies peu représentées restent difficiles. Features ImageNet peuvent rester trop génériques (pas de fine-tuning). Couplage des tâches : une mauvaise modélisation de l'espèce/santé peut impacter la maladie.",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_6_bk.png")
            },
            {
                "num": "7",
                "nom": "Multi-tâche 2 têtes + signal santé",
                "desc": "**Architecture multi-tâche à 2 têtes** : Un backbone CNN pré-entraîné partagé produit un embedding commun. Tête espèce (multi-classe). Tête maladie (multi-classe hors 'healthy', activée uniquement pour échantillons malades). Un signal santé auxiliaire interne est injecté comme feature dans la tête maladie.",
                "workflow": "Phase 1: entraînement des têtes avec backbone gelé (pondérations de pertes, l'échantillon tagué 'healthy' n'entraîne pas la tête maladie). Phase 2: fine-tuning partiel des couches hautes du backbone.",
                "avantages": "Une seule passe backbone pour deux tâches; coût d'inférence réduit. L'injection de la probabilité 'malade' et le masquage de perte évitent que les 'healthy' perturbent la tête maladie. Synergie utile: l'embedding partagé bénéficie des signaux espèce et santé auxiliaire. Equilibre des objectifs via pondérations des pertes.",
                "limites": "Pas de sortie santé explicite: pas de score/label 'healthy vs diseased' livrable tel quel (signal interne non calibré). Dépendance au signal santé: si le signal auxiliaire est biaisé, la tête maladie peut sur- ou sous-activer certaines classes. Conflits d'optimisation: sensibilité aux pondérations et au fine-tuning. Classes rares: malgré le masquage des 'healthy', les maladies peu représentées restent difficiles.",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_7_bk.png")
            },
            {
                "num": "8",
                "nom": "Multi-tâche simplifié",
                "desc": "**Architecture multi-tâche simplifiée (2 têtes)** : Un seul backbone CNN pré-entraîné partagé, et deux têtes parallèles: Espèce, Disease (incluant explicitement healthy). Pas de tête 'santé' dédiée, pas de masquage : toutes les images entraînent les deux têtes.",
                "workflow": "Phase 1: entraînement des têtes avec backbone gelé (pondérations de pertes, label smoothing). Phase 2: fine-tuning partiel du haut du backbone (option gradient clipping). Inférence: une seule passe réseau → deux sorties simultanées: Espèce et Healthy/Maladie.",
                "avantages": "Simplicité: pas de tête santé, pas de règles/mask; supervision uniforme. Efficience: un seul backbone et une seule inférence pour obtenir espèce + santé/maladie. Cohérence de décision: healthy fait partie du même espace que les maladies → seuils et calibration unifiés (softmax à 21 classes). Maintenance légère: pipeline standardisé.",
                "limites": "Déséquilibre 'healthy': la classe healthy peut dominer et biaiser la tête disease_all, au détriment des maladies rares. Pas de conditionnement par espèce: risque de confusions inter-espèces. Seuils globaux: calibration potentiellement sous-optimale pour distributions très différentes selon l'espèce. Shortcut possible: le modèle peut exploiter des corrélations de fond plutôt que des lésions fines.",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_8_bk.png")
            },
            {
                "num": "9",
                "nom": "Architecture conditionnée",
                "desc": "**Architecture conditionnée (Species + Health → Disease)** : Un backbone CNN pré-entraîné unique produit un embedding partagé. Tête espèce (multi-classe). Tête maladie (multi-classe hors 'healthy'), conditionnée par le vecteur de probabilités d'espèce et la probabilité interne d'être malade (tête santé auxiliaire non exposée). Les échantillons 'healthy' n'entraînent pas la tête maladie.",
                "workflow": "Phase 1: apprentissage des têtes avec backbone gelé, pondérations de pertes. Phase 2: fine-tuning partiel des couches hautes. La tête maladie est optimisée uniquement sur les images malades (healthy masqués).",
                "avantages": "Conditionnement explicite: la maladie est guidée par l'info d'espèce et un indicateur de santé, réduisant les confusions inter-espèces et focalisant sur les cas réellement malades. Synergie multi-tâches: l'embedding partagé + signaux auxiliaires apportent un contexte fort. Efficience: un seul backbone; une seule inférence pour obtenir espèce et maladie. Contrôle des compromis via pondérations de pertes.",
                "limites": "Propagation d'erreurs: une erreur d'espèce ou un biais du signal santé peut entraîner une mauvaise prédiction de maladie. Raccourcis/biais: le modèle peut sur-utiliser les a priori espèce/santé au détriment d'indices visuels fins. Pas de sortie santé livrable: la santé est un signal interne. Calibrage sur 'healthy': la tête maladie n'est pas entraînée sur les sains; ses sorties peuvent être peu informatives pour des images réellement 'healthy'.",
                "img": os.path.join(ASSETS_DIR, "architectures/archi_9_bk.png")
            }
        ]

        # Affichage des architectures avec backbone partagé en grille 2xN (2 colonnes par ligne)
        shared_archs = arch_info_shared
        for i in range(0, len(shared_archs), 2):
            row_cols = st.columns(2)
            for col, arch in zip(row_cols, shared_archs[i:i+2]):
                with col:
                    with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                        left_col, right_col = st.columns(2)

                        # Colonne gauche : schéma
                        with left_col:
                            if os.path.exists(arch['img']):
                                st.image(arch['img'], width="stretch")

                        # Zone de détails
                        with st.expander("Plus de détails & Courbe d'apprentissage"):
                            # Courbe de loss (sauf Archi 5)
                            loss_path = os.path.join(ASSETS_DIR, f"architectures/loss_archi_{arch['num']}.png")
                            if arch['num'] != "5" and os.path.exists(loss_path):
                                st.caption(f"Courbe d'apprentissage - Archi {arch['num']}")
                                _render_loss_hover(loss_path, arch['num'])
                                st.divider()
                            
                            st.markdown(f"**Workflow** : {arch['workflow']}")
                            st.markdown(f"**Avantages** : {arch['avantages']}")
                            st.markdown(f"**Limites** : {arch['limites']}")

    # --- STEP 5 ---
    elif current == 5:
        st.subheader("Synthèse des Performances et des Coûts")

        # Chargement des données depuis l'Excel
        excel_path = os.path.join(ASSETS_DIR, "architectures/perfo_archi.xlsx")
        
        if os.path.exists(excel_path):
            try:
                # Lecture avec en-tête sur 2 lignes (gestion des cellules fusionnées)
                df_perf_dl = pd.read_excel(excel_path, header=[0, 1])
                
                # Aplatissement des colonnes MultiIndex
                new_columns = []
                for col in df_perf_dl.columns:
                    c0, c1 = col
                    if str(c0).startswith("Unnamed"):
                        new_columns.append(str(c1).strip())
                    else:
                        new_columns.append(f"{str(c0).strip()}-{str(c1).strip()}")
                df_perf_dl.columns = new_columns
                
                # Formatage de la colonne Archi pour le chart (ex: 1 -> "Archi 1")
                df_chart = df_perf_dl.copy()
                df_chart['Archi_Label'] = df_chart['Archi'].apply(lambda x: f"Archi {x}")

                # Affichage du tableau avec en-tête coloré (si supporté), sans colonne d'index
                styled_df = df_perf_dl.style.set_table_styles([
                    {
                        "selector": "th.col_heading",
                        "props": "background-color: #fdf0d5;"
                    }
                ])
                st.dataframe(styled_df, width="stretch", hide_index=True)

                st.divider()
                
                # Graphique comparatif
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name='Espèce Macro-F1',
                    x=df_chart['Archi_Label'],
                    y=df_chart['Espèce-Macro_F1'],
                    marker_color='lightblue'
                ))
                fig_comp.add_trace(go.Bar(
                    name='Maladie Macro-F1',
                    x=df_chart['Archi_Label'],
                    y=df_chart['Maladie-Macro_F1'],
                    marker_color='lightcoral'
                ))

                # Ajout de la ligne horizontale à Y=0.99
                fig_comp.add_hline(
                    y=0.99,
                    line_dash="dash",
                    line_color="red",
                    line_width=1,
                )

                # Ajout de la ligne horizontale à Y=0.9985 (bleue, plus fine)
                fig_comp.add_hline(
                    y=0.9985,
                    line_dash="dash",
                    line_color="blue",
                    line_width=1,
                )

                fig_comp.update_layout(
                    title="Comparaison des Performances par Architecture",
                    yaxis_range=[0.95, 1.0], # Ajusté car certaines valeurs peuvent varier
                    barmode='group'
                )
                st.plotly_chart(fig_comp, width="stretch")
                
                # Card d'analyse générée
                st.markdown(
                    """
                    <div style="margin-top: 1rem; padding: 1rem; border-radius: 0.5rem; background-color: #e3f2fd; border: 1px solid #90caf9; color: #0d47a1;">
                        <h5 style="margin-top: 0; margin-bottom: 0.5rem;"> Analyse des résultats</h5>
                        <p style="margin-bottom: 0;">
                            Les résultats confirment que le seuil critique de 0.99 de Macro-F1 est dépassé par la majorité des architectures fine-tunées, validant la robustesse du Transfer Learning sur ce dataset.
                            L'Architecture 3 (Unifiée) obtient le meilleur score global sur les maladies (99,34%), illustrant l'efficacité d'un apprentissage conjoint simple des caractéristiques.
                            Enfin, les architectures multi-tâches (notamment la 9) offrent une alternative très compétitive (99,13%) qui maximise l'interprétabilité structurelle sans sacrifier la précision.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier Excel : {e}")
        else:
            st.warning("Fichier de données 'perfo_archi.xlsx' introuvable.")

        st.header("Décisions et Exclusions")
        
        st.markdown("""
        **Architectures exclues :**
        - **Archi 4** : Cascade complexe sans gain tangible, risque de propagation d'erreurs
        - **Archi 6** : En retrait sur la maladie (0.975 vs ≥0.989 pour les autres)
        - **Archi 8** : Pas de bénéfice mesurable vs Archi 7/9
        
        **Architectures retenues pour recommandation :**
        - **Archi 3** : Excellente simplicité de déploiement (1 modèle, 1 inférence)
        - **Archi 7** : Bon compromis performance/efficience
        - **Archi 9** : Conditionnement explicite, synergie maximale
        """)

    # --- STEP 6 ---
    elif current == 6:
        st.markdown("**Interprétabilité (Grad-CAM)**")

        # Top Row: Image + Navigation
        col_proto_img, col_proto_nav = st.columns([0.6, 0.4])

        with col_proto_img:
            # Image fixe demandée
            img_proto = os.path.join(ASSETS_DIR, "exp_gard_cam.png")
            if os.path.exists(img_proto):
                st.image(img_proto, use_container_width=True)
            else:
                st.info("Image 'exp_gard_cam.png' introuvable.")

        with col_proto_nav:
            # Ajout d'une option par défaut pour ne rien afficher au départ
            choice = st.radio(
                " ",
                [
                    "Sélectionnez une analyse...",
                    "1. Pertinence Prédictions",
                    "2. Comparaison Tâches",
                    "3. Hors Contexte Studio"
                ],
            )

        st.divider()

        # Bottom Row: Dynamic Content
        # On n'affiche rien si l'option par défaut est sélectionnée
        if choice != "Sélectionnez une analyse...":
            with st.container():
                if choice == "1. Pertinence Prédictions":
                    st.markdown('<h3 style="margin-bottom: -5rem;">Pertinence des prédictions en phase d’inférence</h3>', unsafe_allow_html=True)
                    
                    c1, c_sep, c2 = st.columns([0.48, 0.04, 0.48])
                    
                    with c1:
                        st.markdown(
                            '<div style="margin-top: -15px;">'
                            "Lorsque le modèle prédit correctement l’espèce et la maladie, les cartes Grad‑CAM montrent‑elles que son attention "
                            "se concentre bien sur la feuille et les lésions pertinentes ?"
                            "</div>",
                            unsafe_allow_html=True
                        )

                        img_path_pred_ok_esp = os.path.join(ASSETS_DIR, "Interpretability/pred_ok_esp.png")
                        if os.path.exists(img_path_pred_ok_esp):
                            st.image(img_path_pred_ok_esp, use_container_width=True, caption="Prédiction correcte - tête espèce")
                            # Card associée à l'image espèce
                            st.markdown(
                                """
                                <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #e3f2fd; border: 1px solid #d0d4e4; color: #0d47a1;">
                                    Les Grad‑CAM montrent des foyers d’attention surtout répartis sur la feuille et les nervures.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        st.markdown(" ")
                        img_path_pred_ok_malad = os.path.join(ASSETS_DIR, "Interpretability/pred_ok_malad.png")
                        if os.path.exists(img_path_pred_ok_malad):
                            st.image(img_path_pred_ok_malad, use_container_width=True, caption="Prédiction correcte - tête maladie")
                            # Card associée à l'image maladie
                            st.markdown(
                                """
                                <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #e3f2fd; border: 1px solid #d0d4e4; color: #0d47a1;">
                                    Les zones chaudes se concentrent majoritairement sur les régions de lésions, décolorations ou bords dégradés ce qui est conforme à l’expertise visuelle attendue.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with c_sep:
                         st.markdown('<div style="border-left: 1px solid #d0d4e4; height: 900px; margin: 0 auto;"></div>', unsafe_allow_html=True)

                    with c2:
                        st.markdown(
                            "Les erreurs viennent‑elles du fait que le modèle regarde ailleurs que les lésions, ou bien qu’il regarde les lésions "
                            "mais se trompe de classe ?"
                        )

                        img_path_err_class = os.path.join(ASSETS_DIR, "Interpretability/err_class.png")
                        if os.path.exists(img_path_err_class):
                            st.image(img_path_err_class, use_container_width=True, caption="Exemple d'erreur de classification")
                        
                        # Card de synthèse spécifique à la colonne droite
                        st.markdown(
                            """
                            <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #e3f2fd; border: 1px solid #d0d4e4; color: #0d47a1;">
                                Les activations sont diffuses sur de larges zones de la feuille (parfois au‑delà des lésions visibles),ce qui suggère que
                                le modèle exploite des motifs globaux de texture/couleur plutôt qu’une localisation très précise des taches. 
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                elif choice == "2. Comparaison Tâches":
                    st.subheader("Comparaison de l’attention entre tâches")
                    st.markdown(
                        """
                        Lorsque l’on passe d’une tâche de classification à l’autre (espèce, maladie), les cartes Grad‑CAM 
                        montrent‑elles un déplacement significatif de l’attention du réseau vers des régions différentes de la feuille ?
                        """
                    )
                    
                    img_path_att = os.path.join(ASSETS_DIR, "Interpretability/attention_réseau.png")
                    if os.path.exists(img_path_att):
                        st.image(img_path_att, use_container_width=True, caption="GRAD-CAM Espèce-maladie")
                    
                    # Card de synthèse sous l'image
                    st.markdown(
                        """
                        <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #e3f2fd; border: 1px solid #d0d4e4; color: #0d47a1;">
                            Les GRAD-CAM mettent en évidence des caractéristiques liées à l’espèce plutôt que les symptômes de la maladie. 
                            Ce résultat s’explique d’une part par l’architecture hiérarchique d’Archi9 et d’autre part par la forte corrélation espèce–maladie 
                            propre au dataset PlantVillage. 
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif choice == "3. Hors Contexte Studio":
                    st.subheader("Hors Contexte Studio")
                    c_bg, c_sep2, c_wild = st.columns([0.48, 0.04, 0.48])
                    
                    with c_bg:
                        st.markdown("**Analyse de l’influence d’une couleur de fond uni différente**")
                        st.markdown(
                            """
                            Les Grad‑CAM montrent‑elles que le modèle s’appuie sur des indices de fond ou de prise de vue caractéristiques du 
                            dataset PlantVillage plutôt que sur des motifs pathologiques réellement liés à la maladie ?
                            """
                        )
                        img_path_fond_esp = os.path.join(ASSETS_DIR, "Interpretability/pre_correct_esp_fond.png")
                        img_path_fond_mala = os.path.join(ASSETS_DIR, "Interpretability/pred_correct_mala_fond.png")

                        if os.path.exists(img_path_fond_esp):
                            st.image(img_path_fond_esp, use_container_width=True, caption="Impact du fond uni - tête espèce")

                        if os.path.exists(img_path_fond_mala):
                            st.image(img_path_fond_mala, use_container_width=True, caption="Impact du fond uni - tête maladie")
                        
                        # Card de synthèse sous les colonnes
                        st.markdown(
                            """
                            <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #e3f2fd; border: 1px solid #d0d4e4; color: #0d47a1;">
                                Ces essais avec un fond saumon suggèrent une dépendance au contexte globalement modérée et variable selon les classes.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with c_sep2:
                         st.markdown('<div style="border-left: 1px solid #d0d4e4; height: 500px; margin: 0 auto;"></div>', unsafe_allow_html=True)

                    with c_wild:
                        st.markdown("**Analyse de l’inférence sur de nouvelles photos « in wild »**")
                        st.markdown(
                            """
                            Quelle est la robustesse de l’attention du modèle en dehors du dataset d’entraînement sur des photos terrain
                            (conditions réelles, non issues de PlantVillage)?
                            """
                        )
                        img_path_in_wild = os.path.join(ASSETS_DIR, "Interpretability/in_wild.png")
                        if os.path.exists(img_path_in_wild):
                            st.image(img_path_in_wild, use_container_width=True, caption="Exemples d'inférence sur photos in the wild")
                        
                        # Card de synthèse sous l'image
                        st.markdown(
                            """
                            <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: #e3f2fd; border: 1px solid #d0d4e4; color: #0d47a1;">
                                Le modèle semble accorder une importance excessive au fond plutôt qu'aux caractéristiques de la feuille. 
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

def sidebar_choice():
    st.title("Deep Learning")
    render_exploration_roadmap()
