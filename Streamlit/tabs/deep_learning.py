import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from utils import render_mermaid


def render_dl_content():
    st.markdown("""
    Le Deep Learning permet d'apprendre automatiquement les features directement à partir des pixels, 
    contrairement au Machine Learning classique qui nécessite une extraction manuelle de descripteurs.
    """)


    
    # --- Phase d'exploration individuelle ---
    with st.expander("👥 Phase d'Exploration Individuelle", expanded=True):
        st.markdown("""
        Dans le cadre de notre formation, **chaque membre de l'équipe a d'abord exploré individuellement 
        un modèle pré-entraîné** pour se familiariser avec les techniques de Deep Learning et comprendre 
        les différents défis liés à :
        """)

        col_img, col_txt = st.columns([1.3, 2])

        with col_img:
            st.image(
                "Streamlit/assets/leviers_DL.png",
                use_container_width=True,
            )

        with col_txt:
            st.markdown("""
            - Le choix du backbone (architecture du réseau)
            - Le fine-tuning et le transfer learning
            - La gestion du déséquilibre des classes
            - L'optimisation des hyperparamètres
            - L'interprétabilité des modèles
            """)

        st.markdown("""
        Cette phase exploratoire nous a permis de **confronter la théorie à la pratique** et d'acquérir 
        une compréhension approfondie des leviers disponibles avant de nous lancer dans l'exploration 
        structurée des 9 architectures.
        """)

        st.markdown("### 🔄 Transfer Learning et Comparaison des Modèles")
        
        st.markdown("""
        Nous avons choisi d'utiliser le **transfert d'apprentissage** car les modèles sont déjà entraînés 
        sur des millions d'images pour détecter des motifs génériques (contours, textures, formes). 
        C'est un **gain de temps et de ressources** considérable.
        """)
        
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
        
        # Transposer pour avoir les modèles en colonnes
        df_models_t = df_models.set_index("Caractéristique").T
        
        st.dataframe(df_models_t, use_container_width=True)
        
        st.success("""
        **🏆 Choix retenu pour l'exploration des architectures : EfficientNetV2S**
        
        EfficientNetV2S offre un **excellent compromis entre performance et efficacité** :
        - **Précision Top-1** de 83,9% sur ImageNet, surpassant ResNet50 (76,1%) et DenseNet-121 (74,4%)
        - **21,5M paramètres** : moins que ResNet50 (25,6M) mais plus que DenseNet-121 (8M)
        - **Efficacité computationnelle** remarquable : latence GPU réduite (5-8 ms)
        - **Précision Top-5** de 96,7%, idéale pour des tâches de classification exigeantes
        - Adapté à nos travaux nécessitant rapidité avec des ressources limitées
        """)

        # Mindmap Théorie du Fine-Tuning
        # Interactive Mindmap Data Definition for Fine-Tuning
        mindmap_ft = {
            "id": "root",
            "text": "Théorie du<br/>Fine-Tuning",
            "children": [
                {
                    "id": "Fondements", 
                    "text": "<b>1. Fondements techniques</b><br/>Transfer Learning",
                    "collapsed": True,
                    "children": [
                        {"id": "Connaissances", "text": "Transfert de connaissances : Exploiter des modèles pré-entraînés sur des millions d'images <i>ImageNet</i>"},
                        {"id": "Motifs", "text": "Détection de motifs génériques : Bords, textures, formes géométriques de base"},
                        {"id": "Efficience", "text": "Gain de temps et de ressources de calcul"}
                    ]
                },
                {
                    "id": "Processus", 
                    "text": "<b>2. Déroulement en 2 Phases</b><br/>Stratégie standard",
                    "collapsed": True,
                    "children": [
                        {
                            "id": "Phase_1", 
                            "text": "Phase 1 : Échauffement<br/><i>Warm-up</i>",
                            "children": [
                                {"id": "Gel", "text": "Backbone gelé : On ne modifie pas les poids de l'extracteur de caractéristiques"},
                                {"id": "Entrainement_Tete", "text": "Entraînement de la tête uniquement : Adaptation aux nouvelles classes"}
                            ]
                        },
                        {
                            "id": "Phase_2", 
                            "text": "Phase 2 : Ajustement fin<br/><i>Fine-tuning</i>",
                            "children": [
                                {"id": "Degel", "text": "Dégel partiel : Déblocage des couches supérieures du backbone"},
                                {"id": "Adaptation", "text": "Spécialisation : Les features génériques deviennent spécifiques aux pathologies végétales"}
                            ]
                        }
                    ]
                },
                {
                    "id": "Parametres", 
                    "text": "<b>3. Paramètres & Mécanismes clés</b>",
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
                    "text": "<b>4. Gains & Bénéfices</b>",
                    "collapsed": True,
                    "children": [
                        {"id": "Performance", "text": "Précision accrue : Atteint des scores >99% inaccessibles sans fine-tuning"},
                        {"id": "Convergence", "text": "Vitesse : Convergence plus rapide qu'un entraînement <i>from scratch</i>"},
                        {"id": "Robustesse", "text": "Adaptation au domaine : Passage de l'image générale à la lésion spécifique"}
                    ]
                },
                {
                    "id": "Risques", 
                    "text": "<b>5. Points de vigilance</b>",
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
    
    # --- Méthodologie ---
    with st.expander("🎯 Méthodologie & Critères de Sélection", expanded=True):
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
        **🎯 Les 3 cas d'usage :**
        - **Cas 1** : Identification d'espèce uniquement
        - **Cas 2** : Diagnostic ciblé (espèce connue → maladie)
        - **Cas 3** : Diagnostic complet (espèce + maladie inconnues)
        """)

    # Onglets principaux DL
    dl_tabs = st.tabs(["🏗️ Architectures", "📊 Performances"])
    
    with dl_tabs[0]:
        st.header("Exploration des 9 Architectures")
        
        st.markdown("""
        **Protocole expérimental commun :**
        - Dataset : PlantVillage/color
        - Backbone pré-entraîné : **EfficientNetV2S** (ImageNet)
        - Splits identiques pour tous les modèles
        - Hyperparamètres fixés : learning rate, batch size, augmentation
        - Métriques : Loss, Accuracy, Macro-F1, matrice de confusion
        """)
        
        st.divider()
        
        # Présentation des architectures
        st.markdown("### 🏗️ Backbone Pré-entraîné Dédié à Chaque Objectif")
        
        arch_info_dedicated = [
            {
                "num": "1",
                "nom": "Trois modèles indépendants",
                "desc": "**Architecture spécialisée** : Trois modèles CNN indépendants, chacun dédié à une seule tâche (species, health, disease). Chaque modèle comprend un backbone pré-entraîné et une tête de classification Dense adaptée au nombre de classes.",
                "workflow": "Chaque modèle s'entraîne en 2 phases sur le même dataset : (1) backbone gelé avec entraînement de la tête uniquement; (2) fine-tuning des dernières couches du backbone pour adapter les features ImageNet aux spécificités du dataset.",
                "avantages": "Simplicité (1 tâche = 1 modèle), absence de conflits entre tâches (pas de compromis dans l'optimisation), performances maximales par tâche (spécialisation totale), interprétabilité facilitée (1 objectif clair par modèle).",
                "limites": "Triplication des ressources (3 backbones à stocker et maintenir), inférences multiples pour cas d'usage complexes, absence de synergie inter-tâches (pas de transfert d'apprentissage entre les 3 têtes), temps d'entraînement cumulé plus long (3 runs).",
                "img": "figures/architectures_dl/archi1.png"
            },
            {
                "num": "2",
                "nom": "Deux modèles (species + disease_extended)",
                "desc": "Deux modèles CNN indépendants : l'un pour l'espèce, l'autre pour l'état sanitaire complet. La classe 'healthy' est intégrée comme une maladie spéciale.",
                "workflow": "Deux runs mono-tâche. Le modèle species s'entraîne sur toutes les images (saines + malades). Le modèle disease_extended s'entraîne également sur toutes les images.",
                "avantages": "Simplicité (2 têtes), uniformité (deux softmax multi-classe), diagnostic complet en 2 inférences (species + disease_extended), 'healthy' est un état sanitaire comme les maladies.",
                "limites": "Déséquilibre accru (classe 'healthy' majoritaire), perte de la métrique binaire explicite healthy/diseased, interprétation plus ambiguë des prédictions mixtes (ex: 40% healthy, 35% early_blight).",
                "img": "figures/architectures_dl/archi2.png"
            },
            {
                "num": "3",
                "nom": "Modèle unifié (35 classes)",
                "desc": "**Architecture unifiée** : Un modèle CNN pré-entraîné + 1 tête Dense softmax (35 classes). Étiquette combinée : chaque image est étiquetée par un couple 'Espèce__État' (Tomato__healthy, Apple__scab…).",
                "workflow": "Phase 1: backbone gelé, entraînement de la tête uniquement. Phase 2: fine-tuning partiel des dernières couches du backbone. Les labels sont pré-combinés en 35 classes.",
                "avantages": "Un seul modèle, une seule inférence : plus simple à déployer et à utiliser. Synergie entre tâches : l'apprentissage capte directement les co-dépendances espèce↔maladie/santé.",
                "limites": "Moins de spécialisation par tâche. Les classes rares peuvent être sous-apprises. Peu flexible : impossible de gérer des paires inédites (nouvelle espèce/maladie) sans réentraîner les 35 classes. Interprétabilité : plus dur d'isoler l'erreur (vient-elle de l'identification d'espèce ou de maladie ?).",
                "img": "figures/architectures_dl/archi3.png"
            },
            {
                "num": "4",
                "nom": "Architecture en cascade",
                "desc": "**Architecture en cascade** : Deux modèles CNN pré-entraînés chaînés. Un classificateur d'espèce extrait un embedding et prédit l'espèce. Un classificateur de maladie global (21 classes, dont 'healthy') reçoit l'image + l'espèce et applique une attention spatiale pour se focaliser sur les zones pertinentes.",
                "workflow": "Phase 1 : backbone gelé, entraînement de la tête. Phase 2 : fine-tuning partiel du backbone. Entraînement du modèle maladie en 2 phases, en lui fournissant l'espèce (True) en entrée pour stabiliser l'apprentissage. Évaluation en CASCADE avec espèce prédite.",
                "avantages": "La prédiction d'espèce guide la maladie, réduisant les confusions entre espèces. L'attention spatiale aide à capter les indices visuels pertinents. Modularité : possibilité d'améliorer séparément espèce ou maladie sans tout réentraîner.",
                "limites": "Une espèce mal prédite dégrade la maladie. Le modèle maladie voit l'espèce (True) à l'entraînement mais la prédite en production. Latence accrue avec passes réseau successives. En cas d'espèce erronée, une maladie impossible peut être proposée.",
                "img": "figures/architectures_dl/archi4.png"
            }
        ]
        
        for arch in arch_info_dedicated:
            with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                col1, col2 = st.columns([1.2, 1])
                
                with col1:
                    st.markdown(f"**Description** : {arch['desc']}")
                    st.markdown(f"**Workflow** : {arch['workflow']}")
                    st.markdown(f"✅ **Avantages** : {arch['avantages']}")
                    st.markdown(f"⚠️ **Limites** : {arch['limites']}")
                
                with col2:
                    if os.path.exists(arch['img']):
                        st.image(arch['img'], caption=f"Schéma Architecture {arch['num']}", use_container_width=True)
        
        st.divider()
        st.markdown("### 🔗 Backbone Pré-entraîné Partagé Entre Plusieurs Objectifs")
        
        arch_info_shared = [
            {
                "num": "5",
                "nom": "CNN + SVM",
                "desc": "**Architecture 'CNN + SVM'** : Un backbone CNN pré-entraîné (gelé) transforme chaque image en vecteur d'embeddings (features). Des classifieurs SVM (espèce, santé, maladie) sont entraînés sur ces embeddings.",
                "workflow": "Sauvegarde des vecteurs + labels. Puis chargement des embeddings, entraînement de trois têtes SVM: Espèce (multi-classe), Santé (binaire: healthy vs diseased), Maladie (soit global multi-classe, soit par espèce).",
                "avantages": "Entraînement très rapide des SVM; itérations légères (on réutilise les embeddings). Simplicité opérationnelle : séparation claire 'features gelées' / 'classifieurs'; facile de remplacer le backbone ou de réentraîner seulement les SVM.",
                "limites": "Les features restent génériques : pas d'adaptation conjointe aux tâches du dataset. Cohérence multi-tâches limitée.",
                "img": "figures/architectures_dl/archi5.png"
            },
            {
                "num": "6",
                "nom": "Multi-tâche unifié (3 têtes)",
                "desc": "**Architecture multi-tâche unifiée** : Un seul backbone CNN pré-entraîné partagé produit un embedding commun, puis trois têtes de classification parallèles: Espèce, Santé, Maladie. La tête 'maladie' est optimisée sur les images malades uniquement.",
                "workflow": "Une seule phase 'têtes seules' avec backbone gelé (pertes pondérées par tête). Pas de fine-tuning activé.",
                "avantages": "Les trois tâches se renforcent (l'espèce et la santé aident la maladie). Un seul backbone à entraîner ; une seule inférence pour obtenir espèce, santé, maladie. Contrôle des compromis via pondérations de pertes par tête.",
                "limites": "Conflits d'optimisation : objectifs parfois concurrents ; sensibilité aux pondérations des pertes. Malgré la tête dédiée, les maladies peu représentées restent difficiles. Features ImageNet peuvent rester trop génériques (pas de fine-tuning). Couplage des tâches : une mauvaise modélisation de l'espèce/santé peut impacter la maladie.",
                "img": "figures/architectures_dl/archi6.png"
            },
            {
                "num": "7",
                "nom": "Multi-tâche 2 têtes + signal santé",
                "desc": "**Architecture multi-tâche à 2 têtes** : Un backbone CNN pré-entraîné partagé produit un embedding commun. Tête espèce (multi-classe). Tête maladie (multi-classe hors 'healthy', activée uniquement pour échantillons malades). Un signal santé auxiliaire interne est injecté comme feature dans la tête maladie.",
                "workflow": "Phase 1: entraînement des têtes avec backbone gelé (pondérations de pertes, l'échantillon tagué 'healthy' n'entraîne pas la tête maladie). Phase 2: fine-tuning partiel des couches hautes du backbone.",
                "avantages": "Une seule passe backbone pour deux tâches; coût d'inférence réduit. L'injection de la probabilité 'malade' et le masquage de perte évitent que les 'healthy' perturbent la tête maladie. Synergie utile: l'embedding partagé bénéficie des signaux espèce et santé auxiliaire. Equilibre des objectifs via pondérations des pertes.",
                "limites": "Pas de sortie santé explicite: pas de score/label 'healthy vs diseased' livrable tel quel (signal interne non calibré). Dépendance au signal santé: si le signal auxiliaire est biaisé, la tête maladie peut sur- ou sous-activer certaines classes. Conflits d'optimisation: sensibilité aux pondérations et au fine-tuning. Classes rares: malgré le masquage des 'healthy', les maladies peu représentées restent difficiles.",
                "img": "figures/architectures_dl/archi7.png"
            },
            {
                "num": "8",
                "nom": "Multi-tâche simplifié",
                "desc": "**Architecture multi-tâche simplifiée (2 têtes)** : Un seul backbone CNN pré-entraîné partagé, et deux têtes parallèles: Espèce, Disease (incluant explicitement healthy). Pas de tête 'santé' dédiée, pas de masquage : toutes les images entraînent les deux têtes.",
                "workflow": "Phase 1: entraînement des têtes avec backbone gelé (pondérations de pertes, label smoothing). Phase 2: fine-tuning partiel du haut du backbone (option gradient clipping). Inférence: une seule passe réseau → deux sorties simultanées: Espèce et Healthy/Maladie.",
                "avantages": "Simplicité: pas de tête santé, pas de règles/mask; supervision uniforme. Efficience: un seul backbone et une seule inférence pour obtenir espèce + santé/maladie. Cohérence de décision: healthy fait partie du même espace que les maladies → seuils et calibration unifiés (softmax à 21 classes). Maintenance légère: pipeline standardisé.",
                "limites": "Déséquilibre 'healthy': la classe healthy peut dominer et biaiser la tête disease_all, au détriment des maladies rares. Pas de conditionnement par espèce: risque de confusions inter-espèces. Seuils globaux: calibration potentiellement sous-optimale pour distributions très différentes selon l'espèce. Shortcut possible: le modèle peut exploiter des corrélations de fond plutôt que des lésions fines.",
                "img": "figures/architectures_dl/archi8.png"
            },
            {
                "num": "9",
                "nom": "Architecture conditionnée",
                "desc": "**Architecture conditionnée (Species + Health → Disease)** : Un backbone CNN pré-entraîné unique produit un embedding partagé. Tête espèce (multi-classe). Tête maladie (multi-classe hors 'healthy'), conditionnée par le vecteur de probabilités d'espèce et la probabilité interne d'être malade (tête santé auxiliaire non exposée). Les échantillons 'healthy' n'entraînent pas la tête maladie.",
                "workflow": "Phase 1: apprentissage des têtes avec backbone gelé, pondérations de pertes. Phase 2: fine-tuning partiel des couches hautes. La tête maladie est optimisée uniquement sur les images malades (healthy masqués).",
                "avantages": "Conditionnement explicite: la maladie est guidée par l'info d'espèce et un indicateur de santé, réduisant les confusions inter-espèces et focalisant sur les cas réellement malades. Synergie multi-tâches: l'embedding partagé + signaux auxiliaires apportent un contexte fort. Efficience: un seul backbone; une seule inférence pour obtenir espèce et maladie. Contrôle des compromis via pondérations de pertes.",
                "limites": "Propagation d'erreurs: une erreur d'espèce ou un biais du signal santé peut entraîner une mauvaise prédiction de maladie. Raccourcis/biais: le modèle peut sur-utiliser les a priori espèce/santé au détriment d'indices visuels fins. Pas de sortie santé livrable: la santé est un signal interne. Calibrage sur 'healthy': la tête maladie n'est pas entraînée sur les sains; ses sorties peuvent être peu informatives pour des images réellement 'healthy'.",
                "img": "figures/architectures_dl/archi9.png"
            }
        ]
        
        for arch in arch_info_shared:
            with st.expander(f"Architecture {arch['num']} : {arch['nom']}"):
                col1, col2 = st.columns([1.2, 1])
                
                with col1:
                    st.markdown(f"**Description** : {arch['desc']}")
                    st.markdown(f"**Workflow** : {arch['workflow']}")
                    st.markdown(f"✅ **Avantages** : {arch['avantages']}")
                    st.markdown(f"⚠️ **Limites** : {arch['limites']}")
                
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
        ### 🎯 Décisions et Exclusions
        
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


def sidebar_choice():
    st.title("🧠 Deep Learning")
    render_dl_content()

