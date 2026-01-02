import json

import streamlit as st
import streamlit.components.v1 as components


def sidebar_choice():
    st.title("🗺️ Mindmap")

    def render_mermaid(mindmap_src: str, height: int = 700):
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
        try {
          svg.setAttribute('width', '100%');
          svg.setAttribute('height', '100%');
          window.panZoom = window.svgPanZoom(svg, {
            zoomEnabled: true,
            controlIconsEnabled: true,
            fit: true,
            center: true,
            minZoom: 0.2,
            maxZoom: 20,
          });
          window.panZoom.resize();
          window.panZoom.fit();
          window.panZoom.center();
        } catch (e) {
          console.error(e);
        }
      }
    </script>
  </body>
</html>
""".replace("__MINDMAP_SRC__", src_json)
        components.html(html, height=height, scrolling=False)

    st.markdown(
        """
Cette page utilise **Markmap** pour rendre une mindmap interactive à partir d'un texte Markdown.

- Tu peux modifier le contenu ci-dessous.
- La mindmap se met à jour automatiquement.
"""
    )

    strategies_md = """---
markmap:
  colorFreezeLevel: 2
---

# Stratégies de Modélisation DL

## Cas 1 : Identification d'espèce

- **Besoin** : Nom de la plante uniquement
- Architecture 1 : Approche Spécialisée
  - Performance maximale : F1-Species **0.9990**
- Architecture 3 : Approche Unifiée
  - Alternative simple et efficace
- Architectures 7 et 9
  - Excellente précision via backbone partagé

## Cas 2 : Diagnostic ciblé

- **Besoin** : Plante connue, cherche la maladie
- Architecture 3 : Rang #1
  - Meilleur F1-Maladie **0.9931**
- Architecture 2 : Hybride
  - Modèle maladie incluant le **"Sain"**
- Architecture 9 : Conditionnée
  - Utilise l'espèce comme signal d'entrée

## Cas 3 : Diagnostic complet

- **Besoin** : Espèce + Santé + Maladie inconnues
- **Architecture 9 : PRODUCTION STANDARD**
  - Meilleur compromis robustesse/précision : F1 **0.9955**
- **Architecture 3 : MOBILE / EDGE**
  - Idéal pour smartphone : **1 seule inférence**
- Architecture 7 : Alternative
  - Excellente performance via signal santé auxiliaire

## Architectures écartées

- **Raisons** : contraintes techniques ou performance
- Architecture 4 : Cascade
  - Rejetée : latence et propagation d'erreurs
- Architecture 6 : Multi-tâche simple
  - Rejetée : performance maladie insuffisante
- Architecture 8 : Multi-tâche 2 têtes
  - Rejetée : moins performante que Archi 7/9
"""

    finetuning_md = """---
markmap:
  colorFreezeLevel: 2
---

# Fine-Tuning Deep Learning

## Principes

- Transfer Learning
- Backbone pré-entraîné
- Hiérarchie des features
  - Couches basses : textures, contours
  - Couches hautes : concepts spécifiques

## Gel / Dégel des couches

- Phase 1 : Backbone gelé
  - Apprentissage des têtes
  - Stabilité + régularisation
- Phase 2 : Dégel partiel
  - Spécialisation domaine
  - Couches hautes
- Phase 3 : Dégel complet (optionnel)
  - Dataset large + régularisation

## Fine-Tuning progressif

- Pourquoi ?
  - Évite catastrophic forgetting
  - Descente de gradient guidée
- Effet
  - Meilleure convergence
  - Apprentissage contrôlé

## Impacts du Fine-Tuning

- Overfitting
  - Gel = régularisation structurelle
  - Réduction de l’espace des hypothèses
- Généralisation
  - Features robustes ImageNet
  - Adaptation sans destruction
- Stabilité des gradients
  - Flux de gradient contrôlé
  - Moins d’oscillations

## Techniques avancées

- Learning rate différencié
  - LR élevé : têtes
  - LR faible : backbone
- Early stopping
  - Surveillance validation
  - Arrêt avant sur-apprentissage
- Régularisation implicite
  - Backbone = prior
  - Contrainte bayésienne implicite

## Message clé soutenance

- Plasticité vs Stabilité
- Fine-Tuning = choix stratégique
"""

    strategies_mermaid = """mindmap
  root((Stratégies de\n    Modélisation DL))
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

    finetuning_mermaid = """mindmap
  root((Fine-Tuning Deep Learning))
    
    Principes
      TransferLearning[\"Transfer Learning\"]
      Backbone[\"Backbone pré-entraîné\"]
      Hierarchie[\"Hiérarchie des features\"]
        Bas[\"Couches basses : textures, contours\"]
        Haut[\"Couches hautes : concepts spécifiques\"]

    Gel_Degel[\"Gel / Dégel des couches\"]
      Phase1[\"Phase 1 : Backbone gelé\"]
        P1Obj[\"Apprentissage des têtes\"]
        P1Effet[\"Stabilité + régularisation\"]
      Phase2[\"Phase 2 : Dégel partiel\"]
        P2Obj[\"Spécialisation domaine\"]
        P2Couches[\"Couches hautes\"]
      Phase3[\"Phase 3 : Dégel complet (optionnel)\"]
        P3Cond[\"Dataset large + régularisation\"]

    Progressif[\"Fine-Tuning progressif\"]
      Pourquoi[\"Pourquoi ?\"]
        Evite[\"Évite catastrophic forgetting\"]
        Guide[\"Descente de gradient guidée\"]
      Effet[\"Effet\"]
        Convergence[\"Meilleure convergence\"]
        Stabilite[\"Apprentissage contrôlé\"]

    Impact[\"Impacts du Fine-Tuning\"]
      Overfit[\"Overfitting\"]
        GelReg[\"Gel = régularisation structurelle\"]
        Reduit[\"Réduction espace hypothèses\"]
      Generalisation[\"Généralisation\"]
        Robustesse[\"Features robustes ImageNet\"]
        Adaptation[\"Adaptation sans destruction\"]
      Gradients[\"Stabilité des gradients\"]
        Flux[\"Flux de gradient contrôlé\"]
        Osc[\"Moins d’oscillations\"]

    Techniques[\"Techniques avancées\"]
      LRdiff[\"Learning rate différencié\"]
        LRhead[\"LR élevé : têtes\"]
        LRback[\"LR faible : backbone\"]
      EarlyStop[\"Early stopping\"]
        Val[\"Surveillance validation\"]
        Stop[\"Arrêt avant sur-apprentissage\"]
      RegImp[\"Régularisation implicite\"]
        Prior[\"Backbone = prior\"]
        Bayes[\"Contrainte bayésienne implicite\"]

    Message[\"Message clé soutenance\"]
      Phrase[\"Plasticité vs Stabilité\"]
      Jury[\"Fine-Tuning = choix stratégique\"]
"""

    render_mode = st.selectbox("Rendu", ["Mermaid (boîtes)", "Markmap (lignes)"])

    if render_mode == "Mermaid (boîtes)":
        templates = {
            "Stratégies de Modélisation DL": strategies_mermaid,
            "Fine-Tuning Deep Learning": finetuning_mermaid,
        }
        template_name = st.selectbox("Mindmap", list(templates.keys()), key="mermaid_template")
        src = st.text_area(
            "Mindmap (Mermaid)",
            value=templates[template_name],
            height=420,
            key=f"mermaid_src_{template_name}",
        )

        render_mermaid(src, height=700)
        return

    try:
        from streamlit_markmap import markmap
    except ModuleNotFoundError:
        st.error(
            "Le package 'streamlit-markmap' n'est pas installé dans cet environnement. "
            "Mets à jour l'environnement puis relance l'app :\n\n"
            "`conda env update -n conda_env -f conda_env.yml`"
        )
        return

    templates = {
        "Stratégies de Modélisation DL": strategies_md,
        "Fine-Tuning Deep Learning": finetuning_md,
    }

    template_name = st.selectbox("Mindmap", list(templates.keys()))
    md = st.text_area(
        "Mindmap (Markdown)",
        value=templates[template_name],
        height=420,
        key=f"mindmap_md_{template_name}",
    )

    markmap(md, height=650)
