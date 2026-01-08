# Scripts de Présentation - HerbI-Dent (Durée cible : ~7 minutes)

Ce document propose une trame détaillée pour une présentation orale d'environ 7 minutes couvrant les pages méthodologiques.

---

## 1. Page "Méthodologie ML - DL" [~1 min 30s]

**Action** : Naviguer vers l'onglet *Méthodologie*.

**Discours Suggéré :**

"Bonjour à tous. Pour répondre au défi complexe de la reconnaissance des plantes et du diagnostic de leurs maladies, nous n'avons pas voulu nous limiter à une seule technique. Nous avons adopté une approche comparative rigoureuse en explorant deux paradigmes fondamentalement différents, que vous voyez illustrés ici.

**(Pointer l'image de gauche - Machine Learning)**
Sur votre gauche, vous avez notre approche **Machine Learning Classique**.
C'est la voie de l'expertise humaine et de l'ingénierie "à la main". Dans cette approche, c'est nous, Data Scientists, qui définissons ce qui est important dans une image. Nous codons des algorithmes pour extraire des formes, des textures, des couleurs. Nous disons à la machine : *"Regarde, ceci est une tache jaune, ceci est une bordure dentelée"*. C'est une démarche très contrôlée, explicable, mais qui demande une connaissance métier pointue et qui peut être limitée si nous, humains, ratons des détails subtils.

**(Pointer l'image de droite - Deep Learning)**
À droite, c'est l'approche **Deep Learning**.
Ici, changement radical de philosophie. Nous ne disons plus au modèle *quoi* chercher. Nous lui donnons les images brutes, pixel par pixel, et nous laissons un réseau de neurones profond (le "cerveau" artificiel) apprendre lui-même ses propres filtres. Il va découvrir seul que la texture de la feuille est importante, ou qu'une certaine nuance de brun signale une maladie. C'est potentiellement beaucoup plus puissant et robuste aux variations, mais c'est aussi plus gourmand en ressources et souvent plus opaque ("boîte noire").

Notre objectif a été de mettre ces deux mondes en compétition pour déterminer quelle approche était la plus viable pour un déploiement réel sur le terrain."

---

## 2. Page "Machine Learning (Roadmap)" [~2 min 30s]

**Action** : Naviguer vers l'onglet *Machine Learning (Roadmap)*.

**Discours Suggéré :**

"Entrons maintenant dans le vif du sujet avec notre pipeline Machine Learning. Pour rendre notre travail lisible et reproductible, nous l'avons structuré en une roadmap séquentielle de 6 étapes clés.

**(Cliquer sur l'étape 1 - Extraction)**
Tout commence par l'**Extraction des Features**.
Une image brute (des milliers de pixels) est inexploitable telle quelle par un algorithme classique comme un Random Forest. Nous avons donc dû réduire cette dimensionnalité tout en gardant l'information essentielle. Nous avons extrait trois types de descripteurs :
1.  **La Couleur** : Via des histogrammes et des moments de couleur (moyenne, écart-type par canal).
2.  **La Texture** : Via les features de Haralick (contraste, homogénéité...), très utiles pour repérer les nécroses sur les feuilles.
3.  **La Forme** : Via les moments de Hu, invariants à la rotation et à l'échelle.

**(Cliquer sur l'étape 2 et 3 - Split & Rééchantillonnage)**
Une fois ces données tabulaires constituées, nous passons à la préparation.
Nous avons d'abord séparé nos données (Train/Test) de manière stricte.
Mais nous avons fait face à un défi majeur : le **déséquilibre des classes**. Certaines maladies sont très rares, d'autres très fréquentes. Si on ne fait rien, le modèle ignore les maladies rares. Nous avons donc utilisé le **SMOTE** (Synthetic Minority Over-sampling Technique) pour générer synthétiquement des exemples des classes minoritaires, mais attention : **uniquement sur le jeu d'entraînement**, pour ne pas biaiser nos tests.

**(Cliquer sur l'étape 4 et 5 - Preprocessing & Modélisation)**
Après avoir mis toutes nos features à la même échelle (StandardScaler), nous avons lancé la compétition.
Nous avons entraîné plusieurs "gladiateurs" : Random Forest, SVM, KNN, XGBoost. Nous avons optimisé leurs hyperparamètres via GridSearch pour chacun d'eux.

**(Cliquer sur l'étape 6 - Évaluation)**
Enfin, le verdict. Nous analysons les résultats non pas juste avec l'Accuracy (trompeuse sur des classes déséquilibrées), mais surtout avec le **F1-Score Macro**, qui nous assure que même les maladies rares sont bien détectées. C'est ce score qui nous servira de référence, ou "Baseline", pour juger si le Deep Learning apporte vraiment une plus-value."

---

## 3. Page "Deep Learning" [~3 min]

**Action** : Naviguer vers l'onglet *Deep Learning*.

**Discours Suggéré :**

"Passons maintenant à l'approche Deep Learning. Ici, la complexité ne réside plus dans l'extraction des features (c'est automatique), mais dans le choix et l'architecture du réseau neuronal.

**(Étape 1 affichée par défaut)**
**Phase 1 : L'Exploration Individuelle**
Avant de converger vers une solution unique, nous avons voulu "sentir" la donnée. Chaque membre de l'équipe a pris un modèle différent (VGG16, ResNet, MobileNet) et a mené ses propres expériences.
Cela a été crucial pour comprendre nos leviers :
*   Jusqu'où peut-on "geler" les couches du réseau ? (Fine-tuning)
*   L'augmentation de données (tourner les images, changer la luminosité) aide-t-elle vraiment à éviter le par cœur (overfitting) ?
Cette étape nous a permis d'harmoniser nos niveaux de compétence et de partager nos échecs et réussites.

**(Cliquer sur l'étape 2 - Démarche Structurée)**
**Phase 2 : Une Méthodologie Orientée Métier**
Nous ne voulions pas faire du Deep Learning "pour la beauté du geste". Nous nous sommes projetés dans un usage réel. Nous avons défini **3 Scénarios d'Usage** précis :
1.  **Le Botaniste** : Qui veut juste identifier l'espèce d'une plante (Cas 1).
2.  **L'Agriculteur Expert** : Qui connait sa tomate mais veut savoir quelle maladie elle a (Cas 2 - Diagnostic ciblé).
3.  **L'Application Grand Public** : L'utilisateur prend une photo et veut tout savoir (Espèce + Maladie) (Cas 3 - Diagnostic complet).

Pour choisir nos modèles, nous avons défini une grille de critères stricte (Figure 14).
Bien sûr, la **Performance (F1-Score)** est reine.
Mais nous avons aussi priorisé l'**Opérabilité** :
*   Le **Coût d'Inférence** (FLOPs) : Est-ce que ça tourne sur un smartphone sans vider la batterie en 30 secondes ?
*   La **Latence** : Est-ce qu'on a la réponse en moins de 100ms ou faut-il attendre 2 secondes ?
*   Le **Poids** : Est-ce que l'utilisateur doit télécharger 500Mo de modèle ?

**(Cliquer sur l'étape 3 - Choix du Backbone)**
**Phase 3 : Le Choix de notre Champion (Backbone)**
Pour nos 9 architectures finales, il nous fallait une colonne vertébrale commune robuste.
Nous avons utilisé le **Transfer Learning** : prendre un modèle qui a déjà vu des millions d'images (ImageNet).
Regardez ce tableau comparatif.
Nous avions ResNet50 (un classique), DenseNet, MobileNet...
Notre choix s'est arrêté sur **EfficientNetV2S**.
Pourquoi ? C'est le meilleur compromis actuel.
*   Il bat ResNet50 en précision (83.9% contre 76.1%).
*   Il est plus léger (21M de paramètres).
*   Et surtout, il est optimisé pour l'inférence rapide.
C'est donc sur cette base solide qu'on a construit nos architectures.

Nous avons aussi établi un protocole expérimental rigoureux : mêmes splits de données pour tout le monde, mêmes hyperparamètres, et un seuil de surveillance de l'overfitting très strict (écart max de 0.5% entre train et val).

Maintenant que les bases sont posées, je vous invite à aller voir les résultats de ces 9 architectures..."

---

**Total estimé : ~7 minutes (en comptant les transitions et les pauses naturelles).**
