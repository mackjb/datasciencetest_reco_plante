import streamlit as st

def sidebar_choice():
    st.title("🏁 Conclusion & Perspectives")
    
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2e7d32;'>
    <b>Synthèse Globale</b> : Nous avons réussi à développer une chaîne de traitement complète, 
    allant de l'analyse exploratoire au développement de modèles de Deep Learning performants.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.header("🏆 Résultats Clés")
        st.success("✅ **Performance** : Le Deep Learning (Archi 9) atteint un F1-score moyen de **99.55%**.")
        st.info("✅ **ML Classique** : Le SVM-RBF reste une excellente baseline avec **93.7%** d'accuracy sur l'espèce.")
        st.warning("⚠️ **Limites** : Biais possible sur le fond (studio) et corrélation espèce-maladie propre au dataset.")
        
    with c2:
        st.header("💼 Impact Métier")
        st.markdown("""
        *   **Gain de temps** : Diagnostic instantané vs expertise humaine coûteuse.
        *   **Scalabilité** : Déploiement possible sur le cloud ou en edge computing.
        *   **Fiabilité** : Standardisation du diagnostic, réduisant l'erreur humaine subjective.
        """)

    st.divider()
    
    st.header("🔮 Perspectives & Améliorations")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.subheader("🔬 Robustesse")
        st.write("Diversifier le dataset avec des fonds variés et des conditions 'Wild' pour casser le biais de studio.")
        
    with col_p2:
        st.subheader("🛠️ Optimisation")
        st.write("Expérimenter des architectures de type **Vision Transformer (ViT)** pour capter des dépendances plus fines.")
        
    with col_p3:
        st.subheader("📱 Déploiement")
        st.write("Utiliser l'**Archi 3** (mono-modèle) pour une intégration fluide sur smartphone via TensorFlow Lite.")

