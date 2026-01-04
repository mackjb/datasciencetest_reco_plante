import json
import streamlit as st
import streamlit.components.v1 as components

from utils import render_mermaid



def sidebar_choice():
    st.title("Conclusion & Perspectives")

    # Mindmap content definition
    # Interactive Mindmap Data Definition
    mindmap_data = {
        "id": "root",
        "text": "Stratégies de<br/>Modélisation DL",
        "children": [
            {
                "id": "Cas_1_Botaniste",
                "text": "<b>Cas 1 : Identification d'espèce</b><br/>Besoin : Nom de la plante uniquement",
                "collapsed": True,
                "children": [
                    {"id": "Archi_1_Expert", "text": "Architecture 1 : Approche Spécialisée<br/>Performance maximale F1-Species 0.9990"},
                    {"id": "Archi_3_Simple", "text": "Architecture 3 : Approche Unifiée<br/>Alternative simple et efficace"},
                    {"id": "Archi_7_9", "text": "Architectures 7 et 9<br/>Excellente précision via backbone partagé"}
                ]
            },
            {
                "id": "Cas_2_Agriculteur",
                "text": "<b>Cas 2 : Diagnostic ciblé</b><br/>Besoin : Plante connue, cherche la maladie",
                "collapsed": True,
                "children": [
                    {"id": "Archi_3_Top", "text": "Architecture 3 : Rang #1<br/>Meilleur F1-Maladie 0.9931"},
                    {"id": "Archi_2_H", "text": "Architecture 2 : Hybride<br/>Modèle maladie incluant le 'Sain'"},
                    {"id": "Archi_9_Context", "text": "Architecture 9 : Conditionnée<br/>Utilise l'espèce comme signal d'entrée"}
                ]
            },
            {
                "id": "Cas_3_Grand_Public",
                "text": "<b>Cas 3 : Diagnostic complet</b><br/>Besoin : Espèce + Santé + Maladie inconnues",
                "collapsed": True,
                "children": [
                    {"id": "Archi_9_PROD", "text": "<b>Architecture 9 : PRODUCTION STANDARD</b><br/>Meilleur compromis robustesse/précision F1 0.9955"},
                    {"id": "Archi_3_MOBILE", "text": "<b>Architecture 3 : MOBILE / EDGE</b><br/>Idéal pour smartphone : 1 seule inférence"},
                    {"id": "Archi_7_Alt", "text": "Architecture 7 : Multi-tâche<br/>Excellente performance via signal santé auxiliaire"}
                ]
            },
            {
                "id": "Architectures_Ecartees",
                "text": "<b>Architectures écartées</b><br/>Raisons techniques ou performance",
                "collapsed": True,
                "children": [
                    {"id": "Archi_4_Cascade", "text": "Architecture 4 : Cascade<br/>Rejetée : Latence et propagation d'erreurs"},
                    {"id": "Archi_6_MT", "text": "Architecture 6 : Multi-tâche simple<br/>Rejetée : Performance maladie insuffisante"},
                    {"id": "Archi_8_MT", "text": "Architecture 8 : Multi-tâche 2 têtes<br/>Rejetée : Moins performante que Archi 7/9"}
                ]
            }
        ]
    }
    
    # Render the interactive mindmap
    render_mermaid(mindmap_data, height=850)

    
    st.divider()
    
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2e7d32;'>
    <b>Synthèse Globale</b> : Nous avons réussi à développer une chaîne de traitement complète, 
    allant de l'analyse exploratoire au développement de modèles de Deep Learning performants.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.header("Résultats Clés")
        st.success("✅ **Performance** : Le Deep Learning (Archi 9) atteint un F1-score moyen de **99.55%**.")
        st.info("✅ **ML Classique** : Le SVM-RBF reste une excellente baseline avec **93.7%** d'accuracy sur l'espèce.")
        st.warning("⚠️ **Limites** : Biais possible sur le fond (studio) et corrélation espèce-maladie propre au dataset.")
        
    with c2:
        st.header("Impact Métier")
        st.markdown("""
        *   **Gain de temps** : Diagnostic instantané vs expertise humaine coûteuse.
        *   **Scalabilité** : Déploiement possible sur le cloud ou en edge computing.
        *   **Fiabilité** : Standardisation du diagnostic, réduisant l'erreur humaine subjective.
        """)

    st.divider()
    
    st.header("Perspectives & Améliorations")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.subheader("Robustesse")
        st.write("Diversifier le dataset avec des fonds variés et des conditions 'Wild' pour casser le biais de studio.")
        
    with col_p2:
        st.subheader("Optimisation")
        st.write("Expérimenter des architectures de type **Vision Transformer (ViT)** pour capter des dépendances plus fines.")
        
    with col_p3:
        st.subheader("Déploiement")
        st.write("Utiliser l'**Archi 3** (mono-modèle) pour une intégration fluide sur smartphone via TensorFlow Lite.")
