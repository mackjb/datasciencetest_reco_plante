import streamlit as st
import pandas as pd

def render_exploration_roadmap():
    # Session state key for the consolidated page
    if 'dl_step' not in st.session_state:
        st.session_state['dl_step'] = 1

    # STEPS: Titles from MIX Version (which are from New Version)
    steps = {
        1: "Phase d'Exploration Individuelle",
        2: "Démarche Structurée & Critères",
        3: "Exploration d'architectures DL"
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
    cols = st.columns([2, 0.5, 2, 0.5, 2])
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
        st.header("1. Phase d'Exploration Individuelle")
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
        st.header("2. Démarche Structurée & Critères")
        
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
        st.header("3. Exploration d'architectures DL")
        
        st.info("Pourquoi le Transfer Learning ? Pour profiter de modèles entraînés sur des millions d'images (ImageNet).")

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
        st.info("Nous avons conçu **9 Architectures** différentes pour répondre à ces cas d'usage, que nous vous proposons de découvrir dès maintenant.")

def sidebar_choice():
    st.title("Deep Learning")
    render_exploration_roadmap()
