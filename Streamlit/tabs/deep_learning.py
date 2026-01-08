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


def render_dl_content():
    st.markdown("""
    Cette section détaille les **9 architectures** que nous avons conçues et testées.
    L'objectif était de comparer différentes stratégies (mono-modèle, multi-tâches, cascade) pour répondre aux 3 cas d'usage métier.
    """)

    # --- Méthodologie ---
    with st.expander("Rappel : Méthodologie & Critères", expanded=False):
        st.markdown("""
        ### Démarche structurée en 3 étapes :
        
        1. **Exploration** : 9 architectures testées pour comprendre le Deep Learning et ses défis
        2. **Évaluation comparative** : Restriction à quelques architectures couvrant 3 cas d'usage
        3. **Sélection & Recommandation** : Projection pour un déploiement réel
        
        ### Critères de sélection :
        
        | Catégorie | Critères | Justification |
        |-----------|----------|---------------|
        | **Métier** | Précision (Macro-F1, Accuracy) | Capacité à bien prédire toutes les classes |
        | | Généralisation (écart val/test) | Robustesse du modèle (<2% = bon, >5% = overfitting) |
        | | Couverture opérationnelle | Réponse aux 3 cas d'usage métier |
        | **Technique** | Coût d'inférence (FLOPs, latence) | Impact sur batterie et expérience utilisateur |
        | | Coût d'entraînement (temps, GPU) | Budget cloud et itérations rapides |
        | | Complexité (paramètres, maintenabilité) | Taille du modèle et facilité de maintenance |
        | **Autres** | Interprétabilité | Capacité à expliquer les prédictions (Grad-CAM) |
        | | Besoins en données | Quantité d'images annotées nécessaire |
        """)
        
        st.info("""
        **Les 3 cas d'usage :**
        **Les 3 cas d'usage :**
        - **Cas 1** : Identification d'espèce uniquement
        - **Cas 2** : Diagnostic ciblé (espèce connue → maladie)
        - **Cas 3** : Diagnostic complet (espèce + maladie inconnues)
        """)

    # Onglets principaux DL
    dl_tabs = st.tabs(["Architectures", "Performances"])
    
    with dl_tabs[0]:
        st.header("Protocole expérimental commun pour les 9 Architectures")
        
        st.markdown("""
        - Dataset : PlantVillage/color
        - Backbone pré-entraîné : **EfficientNetV2S** (ImageNet)
        - Splits identiques pour tous les modèles
        - Hyperparamètres fixés : learning rate, batch size, augmentation
        - Métriques : Loss, Accuracy, Macro-F1, matrice de confusion
        """)

        st.divider()
        
        # Présentation des architectures
        st.markdown("### Backbone Pré-entraîné Dédié à Chaque Objectif")
        
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
        
        for arch in arch_info_dedicated:
            with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                col1, col2 = st.columns([1.2, 1])
                
                with col1:
                    st.markdown(f"**Description** : {arch['desc']}")
                    st.markdown(f"**Workflow** : {arch['workflow']}")
                    st.markdown(f"**Avantages** : {arch['avantages']}")
                    st.markdown(f"**Limites** : {arch['limites']}")
                
                with col2:
                    if os.path.exists(arch['img']):
                        st.image(arch['img'], caption=f"Schéma Architecture {arch['num']}", use_container_width=True)
        
        st.divider()
        st.markdown("### Backbone Pré-entraîné Partagé Entre Plusieurs Objectifs")
        
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
        
        for arch in arch_info_shared:
            with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                col1, col2 = st.columns([1.2, 1])
                
                with col1:
                    st.markdown(f"**Description** : {arch['desc']}")
                    st.markdown(f"**Workflow** : {arch['workflow']}")
                    st.markdown(f"**Avantages** : {arch['avantages']}")
                    st.markdown(f"**Limites** : {arch['limites']}")
                
                with col2:
                    if os.path.exists(arch['img']):
                        st.image(arch['img'], caption=f"Schéma Architecture {arch['num']}", use_container_width=True)
    
    with dl_tabs[1]:
        st.header("Synthèse des Performances")

        
        # Tableau de performances
        perf_dl = {
            "Architecture": ["Archi 1", "Archi 2", "Archi 3", "Archi 4", "Archi 5", "Archi 6", "Archi 7", "Archi 8", "Archi 9"],
            "Species Macro-F1": [0.990, 0.990, 0.990, 0.990, 0.985, 0.988, 0.990, 0.989, 0.990],
            "Species Accuracy": [0.990, 0.990, 0.990, 0.990, 0.986, 0.988, 0.990, 0.989, 0.990],
            "Disease Accuracy": [0.990, 0.988, 0.990, 0.987, 0.982, 0.975, 0.990, 0.986, 0.990],
            "FLOPs (relatif)": ["3×", "2×", "1×", "2×", "1×", "1×", "1×", "1×", "1×"],
            "Maintenabilité": ["Faible", "Moyenne", "Élevée", "Faible", "Moyenne", "Moyenne", "Moyenne", "Moyenne", "Faible"]
        }
        df_perf_dl = pd.DataFrame(perf_dl)
        
        st.dataframe(df_perf_dl.style.highlight_max(subset=["Species Macro-F1", "Species Accuracy", "Disease Accuracy"], axis=0))
        
        st.divider()
        
        st.markdown("""
        ### Décisions et Exclusions
        
        **Architectures exclues :**
        - **Archi 4** : Cascade complexe sans gain tangible, risque de propagation d'erreurs
        - **Archi 6** : En retrait sur la maladie (0.975 vs ≥0.989 pour les autres)
        - **Archi 8** : Pas de bénéfice mesurable vs Archi 7/9
        
        **Architectures retenues pour recommandation :**
        - **Archi 3** : Excellente simplicité de déploiement (1 modèle, 1 inférence)
        - **Archi 7** : Bon compromis performance/efficience
        - **Archi 9** : Conditionnement explicite, synergie maximale
        """)
        
        # Graphique comparatif
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Species Macro-F1',
            x=df_perf_dl['Architecture'],
            y=df_perf_dl['Species Macro-F1'],
            marker_color='lightblue'
        ))
        fig_comp.add_trace(go.Bar(
            name='Disease Accuracy',
            x=df_perf_dl['Architecture'],
            y=df_perf_dl['Disease Accuracy'],
            marker_color='lightcoral'
        ))
        fig_comp.update_layout(
            title="Comparaison des Performances par Architecture",
            yaxis_range=[0.97, 1.0],
            barmode='group'
        )
        st.plotly_chart(fig_comp, use_container_width=True)

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

        st.divider()

        # Mindmap Théorie du Fine-Tuning
        # Interactive Mindmap Data Definition for Fine-Tuning
        mindmap_ft = {
            "id": "root",
            "text": "Théorie du Fine-Tuning",
            "children": [
                {
                    "id": "Fondements", 
                    "text": "1. Fondements techniques<br/>Transfer Learning",
                    "collapsed": True,
                    "children": [
                        {"id": "Connaissances", "text": "Transfert de connaissances : Exploiter des modèles pré-entraînés sur des millions d'images ImageNet"},
                        {"id": "Motifs", "text": "Détection de motifs génériques : Bords, textures, formes géométriques de base"},
                        {"id": "Efficience", "text": "Gain de temps et de ressources de calcul"}
                    ]
                },
                {
                    "id": "Processus", 
                    "text": "2. Déroulement en 2 Phases<br/>Stratégie standard",
                    "collapsed": True,
                    "children": [
                        {
                            "id": "Phase_1", 
                            "text": "Phase 1 : Échauffement<br/>Warm-up",
                            "children": [
                                {"id": "Gel", "text": "Backbone gelé : On ne modifie pas les poids de l'extracteur de caractéristiques"},
                                {"id": "Entrainement_Tete", "text": "Entraînement de la tête uniquement : Adaptation aux nouvelles classes"}
                            ]
                        },
                        {
                            "id": "Phase_2", 
                            "text": "Phase 2 : Ajustement fin<br/>Fine-tuning",
                            "children": [
                                {"id": "Degel", "text": "Dégel partiel : Déblocage des couches supérieures du backbone"},
                                {"id": "Adaptation", "text": "Spécialisation : Les features génériques deviennent spécifiques aux pathologies végétales"}
                            ]
                        }
                    ]
                },
                {
                    "id": "Parametres", 
                    "text": "3. Paramètres & Mécanismes clés",
                    "collapsed": True,
                    "children": [
                        {"id": "Backbone", "text": "Backbone pré-entraîné : ex. EfficientNetV2-S choisi pour son équilibre performance/latence"},
                        {"id": "Learning_Rate", "text": "Learning Rate réduit : Indispensable en Phase 2 pour ne pas détruire les poids pré-entraînés"},
                        {"id": "Regularisation", "text": "Label Smoothing & Gradient Clipping : Stabilisation de l'apprentissage"},
                        {"id": "Optimisation", "text": "Poids de classes : Pour gérer le déséquilibre du dataset"}
                    ]
                },
                {
                    "id": "Gains", 
                    "text": "4. Gains & Bénéfices",
                    "collapsed": True,
                    "children": [
                        {"id": "Performance", "text": "Précision accrue : Atteint des scores >99% inaccessibles sans fine-tuning"},
                        {"id": "Convergence", "text": "Vitesse : Convergence plus rapide qu'un entraînement from scratch"},
                        {"id": "Robustesse", "text": "Adaptation au domaine : Passage de l'image générale à la lésion spécifique"}
                    ]
                },
                {
                    "id": "Risques", 
                    "text": "5. Points de vigilance",
                    "collapsed": True,
                    "children": [
                        {"id": "Overfitting", "text": "Sur-apprentissage : Si le modèle mémorise au lieu de généraliser"},
                        {"id": "Features_Generiques", "text": "Features trop génériques : Si le fine-tuning n'est pas activé, la perte peut être catastrophique"},
                        {"id": "Desequilibre", "text": "Déséquilibre des classes : Biais vers les classes majoritaires"},
                        {"id": "Fragilite_Wild", "text": "Fragilité 'In-Wild' : Chute de performance hors fond uniforme"}
                    ]
                }
            ]
        }
        
        render_mermaid(mindmap_ft, height=600)
        st.caption("Figure : Principes et stratégie de Fine-Tuning appliqués au projet")

    # Onglet Interprétabilité (Grad-CAM)
    with dl_tabs[2]:
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
    st.title("Deep Learning - Architectures")
    render_dl_content()
