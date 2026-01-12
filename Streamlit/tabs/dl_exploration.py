import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import base64
from utils import render_mermaid

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
        1: "Phase d'Exploration Individuelle",
        2: "Démarche Structurée & Critères",
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
        st.header("Phase d'Exploration Individuelle")
        st.markdown("""
        Dans le cadre de notre formation, **chaque membre de l'équipe a d'abord exploré individuellement 
        un modèle pré-entraîné** pour se familiariser avec les techniques de Deep Learning.
        """)
        
        col_img, col_txt = st.columns([1.3, 2])
        with col_img:
            st.image("Streamlit/assets/leviers_DL.png", use_container_width=True)
        with col_txt:
            st.markdown("### Objectifs & Défis")
            st.markdown("""
            Nous avons chacun travaillé sur des notebooks séparés pour comprendre les impacts de :
            - **Le choix du backbone** (VGG, ResNet, EfficientNet...)
            - **Le Fine-Tuning** : Gel partiel vs total des couches.
            - **L'Augmentation de données** : Impact sur l'overfitting.
            - **Le Déséquilibre des classes** : Utilisation de class_weights.
            
            Cette étape était cruciale pour **harmoniser nos connaissances** avant de définir une architecture commune.
            """)

    # --- STEP 2 ---
    elif current == 2:
        st.header("Démarche Structurée & Critères")
        
        st.markdown("""
        Pour structurer notre approche, nous avons défini **3 cas d'usage** correspondant à différents niveaux de complexité métier.
        """)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Cas 1", "Espèce Uniquement")
            st.caption("Identifier la plante avant de chercher la maladie.")
        with c2:
            st.metric("Cas 2", "Diagnostic Ciblé")
            st.caption("On connait l'espèce -> Quelle est la maladie ?")
        with c3:
            st.metric("Cas 3", "Diagnostic Complet")
            st.caption("Image brute -> Espèce + Maladie (Inconnus).")
            
        st.markdown("### Critères de Sélection des Architectures")
        st.table(pd.DataFrame({
            "Catégorie": ["Métier", "Métier", "Technique", "Technique", "Autres"],
            "Critère": ["Précision (Macro-F1)", "Généralisation", "Coût Inférence", "Complexité", "Interprétabilité"],
            "Importance": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐⭐⭐"]
        }))

    # --- STEP 3 ---
    elif current == 3:
        st.header("Transfer Learning")
        

        st.info("Pourquoi le Transfer Learning ? Pour profiter de modèles entraînés sur des millions d'images (ImageNet).")
        
        # Mindmap Théorie du Transfer Learning
        mindmap_tl = {
            "id": "root",
            "text": "Théorie du<br/>Transfer Learning",
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
                        {"id": "EfficientNet", "text": "<b>EfficientNetV2-S</b> : Choisi pour son équilibre Performance/Latence (83.9% Top-1 ImageNet)"},
                        {"id": "Classiques", "text": "ResNet50, DenseNet-121, VGG16 : Modèles standards pour l'extraction de motifs"},
                        {"id": "ViT", "text": "Vision Transformers (ViT) : Alternative non-convolutive gérant les relations globales"}
                    ]
                },
                {
                    "id": "Limites", 
                    "text": "<b>5. Limites & Vigilance</b>",
                    "collapsed": True,
                    "children": [
                        {"id": "Biais_Contexte", "text": "Sensibilité au fond uniforme de PlantVillage (Biais de studio)"},
                        {"id": "Generalisation", "text": "Chute de performance (-5% à -15%) face aux conditions réelles 'In-Wild'"},
                        {"id": "Correlation", "text": "Risque d'apprendre des raccourcis Espèce <-> Maladie plutôt que les lésions"}
                    ]
                }
            ]
        }
        render_mermaid(mindmap_tl, height=600)
        st.divider()



        st.subheader("Choix du Backbone Pré-entraîné")
        st.markdown("**Comparatif des Modèles Pré-entraînés Explorés :**")
        
        models_comparison = {
            "Caractéristique": ["Année", "Auteurs/Org", "Paramètres (M)", "Taille modèle (MB)", 
                               "GFLOPs (224×224)", "GFLOPs (256×256)", "Taille vecteur sortie",
                               "Top-1 Acc ImageNet", "Top-5 Acc ImageNet", "Latence CPU (ms)", 
                               "Latence GPU (ms)", "Taille entrée", "Profondeur (layers)"],
            "EfficientNetV2-S": [2021, "Google Brain", 21.5, "~86", 8.4, "~10.8", 1280, 
                                "83.9%", "96.7%", "60-80", "5-8", "384×384 (optim.)", "~150"],
            "ResNet50": [2015, "Microsoft Research", 25.6, "~102", 4.1, "~5.3", 2048,
                        "76.1%", "93.0%", "40-50", "3-5", "224×224", "50"],
            "YOLOv8n-cls*": [2023, "Ultralytics", 2.7, "~11", 4.2, "~5.4", 1024,
                           "69.0%", "88.3%", "25-35", "2-4", "224×224", "~100"],
            "DenseNet-121": [2017, "Cornell/Facebook", 8.0, "~32", 2.9, "~3.7", 1024,
                           "74.4%", "92.0%", "30-40", "3-5", "224×224", "121"]
        }
        df_models = pd.DataFrame(models_comparison)
        st.dataframe(df_models.set_index("Caractéristique").T, use_container_width=True)
        
        st.success("""
        **Choix retenu : EfficientNetV2S**
        
        - **Performance** : Meilleure Accuracy ImageNet (83.9%) parmi les modèles testés.
        - **Efficience** : Excellent ratio performance/paramètres (21.5M).
        - **Rapidité** : Optimisé pour une inférence rapide sur GPU.
        """)

        st.divider()
        
        st.divider()
        st.divider()
        st.info("Nous avons conçu **9 Architectures** différentes pour répondre à ces cas d'usage, que nous vous proposons de découvrir dès maintenant.")

    # --- STEP 4 ---
    elif current == 4:
        st.header("Architectures")
        
        st.header("Protocole expérimental commun pour les 9 Architectures")
        
        st.markdown("""
        - Dataset : PlantVillage/color
        - Backbone pré-entraîné : **EfficientNetV2S** (ImageNet)
        - Splits identiques pour tous les modèles
        - Hyperparamètres fixés : learning rate, batch size, augmentation
        - Métriques : Loss, Accuracy, Macro-F1, matrice de confusion
        """)

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
                with st.expander(f"Architecture {arch['num']} : {arch['nom']}", expanded=True):
                    # Schéma (Pleine largeur)
                    if os.path.exists(arch['img']):
                        st.image(arch['img'], use_container_width=True)
                    
                    # Phrase de description (sous l'image)
                    st.markdown(f"**Description** : {arch['desc']}")

                    # Zone de détails (Workflow, Avantages, Limites + Loss)
                    with st.expander("Plus de détails & Courbe d'apprentissage"):
                        # Courbe de loss
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
                with st.expander(f"Architecture {arch['num']} : {arch['nom']}", expanded=True):
                    # Schéma (Pleine largeur)
                    if os.path.exists(arch['img']):
                        st.image(arch['img'], use_container_width=True)
                    
                    # Phrase de description (sous l'image)
                    st.markdown(f"**Description** : {arch['desc']}")

                    # Zone de détails (Workflow, Avantages, Limites + Loss)
                    with st.expander("Plus de détails & Courbe d'apprentissage"):
                        # Courbe de loss
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
                    with st.expander(f"Architecture {arch['num']} : {arch['nom']}", expanded=True):
                        # Schéma (Pleine largeur)
                        if os.path.exists(arch['img']):
                            st.image(arch['img'], use_container_width=True)
                        
                        # Phrase de description
                        st.markdown(f"**Description** : {arch['desc']}")

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
        st.header("Performances")

        # Chargement des données depuis l'Excel
        excel_path = os.path.join(ASSETS_DIR, "architectures/perfo_archi.xlsx")
        
        if os.path.exists(excel_path):
            try:
                df_perf_dl = pd.read_excel(excel_path)
                
                # Formatage de la colonne Archi pour le chart (ex: 1 -> "Archi 1")
                df_chart = df_perf_dl.copy()
                df_chart['Archi_Label'] = df_chart['Archi'].apply(lambda x: f"Archi {x}")

                # Affichage du tableau
                # Highlight des métriques principales
                st.dataframe(df_perf_dl.style.highlight_max(
                    subset=["Espèce-Macro_F1", "Espèce-Accuracy", "Maladie- Accuracy"], 
                    axis=0, color='#d1e7dd'
                ), use_container_width=True)

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
                    name='Maladie Accuracy',
                    x=df_chart['Archi_Label'],
                    y=df_chart['Maladie- Accuracy'],
                    marker_color='lightcoral'
                ))
                fig_comp.update_layout(
                    title="Comparaison des Performances par Architecture",
                    yaxis_range=[0.90, 1.0], # Ajusté car certaines valeurs peuvent varier
                    barmode='group'
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
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
        st.header("Interprétabilité (Grad-CAM)")

        # 1) Pertinence des prédictions en phase d'inférence
        with st.expander("Pertinence des prédictions du modèle en phase d’inférence"):
            st.markdown(
                """
                Cette section présente des exemples d’images correctement / incorrectement classées
                avec leur carte Grad-CAM associée, afin de vérifier si le modèle se focalise bien sur
                les lésions et zones pertinentes lors de l’inférence.
                """
            )

            col_esp, col_malad, col_err_class = st.columns([2,2,3])

            img_path_pred_ok_esp = os.path.join(ASSETS_DIR, "Interpretability/pred_ok_esp.png")
            img_path_pred_ok_malad = os.path.join(ASSETS_DIR, "Interpretability/pred_ok_malad.png")
            img_path_err_class = os.path.join(ASSETS_DIR, "Interpretability/err_class.png")

            with col_esp:
                if os.path.exists(img_path_pred_ok_esp):
                    st.image(img_path_pred_ok_esp, use_container_width=True, caption="Prédiction correcte - tête espèce")

            with col_malad:
                if os.path.exists(img_path_pred_ok_malad):
                    st.image(img_path_pred_ok_malad, use_container_width=True, caption="Prédiction correcte - tête maladie")

            with col_err_class:
                if os.path.exists(img_path_err_class):
                    st.image(img_path_err_class, use_container_width=True, caption="Exemple d'erreur de classification")
            
        st.divider()

        # 2) Comparaison de l’attention entre tâches
        with st.expander("Comparaison de l’attention du réseau entre les tâches de classification"):
            st.markdown(
                """
                Ici sont comparées les cartes Grad-CAM obtenues pour différentes têtes de
                classification (espèce, santé, maladie) afin d’illustrer comment l’attention
                du réseau varie selon la tâche optimisée.
                """
            )
            img_path_err_class = os.path.join(ASSETS_DIR, "Interpretability/attention_réseau.png")
            if os.path.exists(img_path_err_class):
                st.image(img_path_err_class, use_container_width=True, caption="GRAD-CAM Espèce-maladie")

        st.divider()

        # 3) Influence d'une couleur de fond différente
        with st.expander("Analyse de l’influence d’une couleur de fond uni différente"):
            st.markdown(
                """
                Des expériences de sensibilité au fond (fond noir vs fond uni coloré, etc.)
                permettent de visualiser l’impact du background sur les activations Grad-CAM
                et de mettre en évidence d’éventuels raccourcis pris par le modèle.
                """
            )
            img_path_fond_esp = os.path.join(ASSETS_DIR, "Interpretability/pre_correct_esp_fond.png")
            img_path_fond_mala = os.path.join(ASSETS_DIR, "Interpretability/pred_correct_mala_fond.png")

            col_fond_esp, col_fond_mala = st.columns(2)

            with col_fond_esp:
                if os.path.exists(img_path_fond_esp):
                    st.image(img_path_fond_esp, use_container_width=True, caption="Impact du fond uni - tête espèce")

            with col_fond_mala:
                if os.path.exists(img_path_fond_mala):
                    st.image(img_path_fond_mala, use_container_width=True, caption="Impact du fond uni - tête maladie")

        st.divider()

        # 4) Inférence sur des images "in the wild"
        with st.expander("Analyse de l’inférence sur de nouvelles photos « in wild »"):
            st.markdown(
                """
                Enfin, cette partie montre des résultats Grad-CAM sur des photos terrain
                (conditions réelles, non issues de PlantVillage) pour évaluer la robustesse
                de l’attention du modèle en dehors du dataset d’entraînement.
                """
            )
            img_path_in_wild = os.path.join(ASSETS_DIR, "Interpretability/in_wild.png")
            if os.path.exists(img_path_in_wild):
                st.image(img_path_in_wild, use_container_width=True, caption="Exemples d'inférence sur photos in the wild")

def sidebar_choice():
    st.title("Deep Learning")
    render_exploration_roadmap()
