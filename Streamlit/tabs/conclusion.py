import json
import streamlit as st
import streamlit.components.v1 as components

def render_mermaid(mindmap_src: str, height: int = 700):
    """
    Renders a Mermaid mindmap with custom CSS and pan/zoom functionality.
    Includes fixes for truncation and layout stability.
    """
    src_json = json.dumps(mindmap_src)
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body {
        margin: 0;
        padding: 0;
        background: transparent;
        height: 100%;
        width: 100%;
      }
      #container {
        width: 100%;
        height: 100vh;
        overflow: hidden;
      }
      .mermaid {
        width: 100%;
        height: 100%;
      }
      /* Critical fix: Override max-width to prevent truncation */
      .mermaid svg {
        max-width: none !important;
        width: 100% !important;
        height: 100% !important;
      }
    </style>
  </head>
  <body>
    <div id="container">
      <div class="mermaid" id="mermaid"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>

    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme: 'base',
        themeVariables: {
          fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
        },
      });

      const src = __MINDMAP_SRC__;
      const el = document.getElementById('mermaid');
      el.textContent = src;

      try {
        await mermaid.run({ nodes: [el] });
      } catch (e) {
        console.error(e);
      }

      const svg = document.querySelector('#container svg');
      if (svg && window.svgPanZoom) {
        // Delay initialization to ensure rendering is complete
        setTimeout(() => {
          try {
            // Explicitly force style overrides in JS as a backup
            svg.style.maxWidth = 'none';
            svg.setAttribute('width', '100%');
            svg.setAttribute('height', '100%');

            window.panZoom = window.svgPanZoom(svg, {
              zoomEnabled: true,
              controlIconsEnabled: true,
              fit: true,
              center: true,
              minZoom: 0.1,
              maxZoom: 10,
            });
            
            // Re-center on window resize
            window.addEventListener('resize', function(){
              window.panZoom.resize();
              window.panZoom.fit();
              window.panZoom.center();
            });

            // Initial view setup with zoom
            window.panZoom.resize();
            window.panZoom.fit();
            window.panZoom.center();
            window.panZoom.zoomBy(2.0);
          } catch (e) {
            console.error(e);
          }
        }, 500);
      }
    </script>
  </body>
</html>
""".replace("__MINDMAP_SRC__", src_json)
    components.html(html, height=height, scrolling=False)


def sidebar_choice():
    st.title("Conclusion & Perspectives")

    # Mindmap content definition
    mindmap_source = """mindmap
  root((Stratégies de
    Modélisation DL))
    Cas_1_Botaniste(<b>Cas 1 : Identification d'espèce</b><br/>Besoin : Nom de la plante uniquement)
      Archi_1_Expert(Architecture 1 : Approche Spécialisée<br/>Performance maximale F1-Species 0.9990)
      Archi_3_Simple(Architecture 3 : Approche Unifiée<br/>Alternative simple et efficace)
      Archi_7_9(Architectures 7 et 9<br/>Excellente précision via backbone partagé)
    
    Cas_2_Agriculteur(<b>Cas 2 : Diagnostic ciblé</b><br/>Besoin : Plante connue, cherche la maladie)
      Archi_3_Top(Architecture 3 : Rang #1<br/>Meilleur F1-Maladie 0.9931)
      Archi_2_H(Architecture 2 : Hybride<br/>Modèle maladie incluant le 'Sain')
      Archi_9_Context(Architecture 9 : Conditionnée<br/>Utilise l'espèce comme signal d'entrée)
    
    Cas_3_Grand_Public(<b>Cas 3 : Diagnostic complet</b><br/>Besoin : Espèce + Santé + Maladie inconnues)
      Archi_9_PROD(<b>Architecture 9 : PRODUCTION STANDARD</b><br/>Meilleur compromis robustesse/précision F1 0.9955)
      Archi_3_MOBILE(<b>Architecture 3 : MOBILE / EDGE</b><br/>Idéal pour smartphone : 1 seule inférence)
      Archi_7_Alt(Architecture 7 : Multi-tâche<br/>Excellente performance via signal santé auxiliaire)

    Architectures_Ecartees(<b>Architectures écartées</b><br/>Raisons techniques ou performance)
      Archi_4_Cascade(Architecture 4 : Cascade<br/>Rejetée : Latence et propagation d'erreurs)
      Archi_6_MT(Architecture 6 : Multi-tâche simple<br/>Rejetée : Performance maladie insuffisante)
      Archi_8_MT(Architecture 8 : Multi-tâche 2 têtes<br/>Rejetée : Moins performante que Archi 7/9)
"""
    
    # Render the mindmap with specified logic
    render_mermaid(mindmap_source, height=850)
    
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
