# TODO

JBM : JB
SL : Lionel
MP : Morgan
BG : Bernadette


dataframe à la granularité Classe
- [X] Nombre de classification, images par classe pour plantvillage dans un df
- [SL] Nettoyer le graphique histograme pour lister les effectis par classe plantvillage
- [SL] Compléter le df plantvillage avec les résolution et les extension (.jpg .png ...)

dataframe à la granularité Images
- [MP] Parcourir toutes les images (segmentées) et compter les pixel différentes du background (noir plantvillage, blanc flavia)
- [] Pourcentage des pixels non background sur la bordure 1 pixel de l'image 
- [] Choisir une métrique de détection d'exposition de la photo, Ratio de pixels “clippés” sur une image, Moyenne et écart-type de la luminance, Entropie de l’histogramme, Métriques no-reference (NR-IQA)
- [JBM] Extraire les segments avec SAM et confronter les métriques plus haut 



🎯 Objectif général du projet :

Utiliser le Machine Learning pour classifier efficacement des espèces végétales (Flavia) et détecter des maladies (PlantVillage) à partir d’images, en réalisant une exploration rigoureuse des données, une visualisation pertinente, un prétraitement adapté, et un feature engineering efficace.

⸻

🚧 Backlog projet ML (Dataset Flavia & PlantVillage)

✅ Partie 1 : Préparation & Prétraitement des données
	•	1.1 Acquisition et structuration des données
	•	Télécharger et vérifier les datasets Flavia et PlantVillage.
	•	Créer une structure claire des dossiers (par espèce et maladie).
	•	Évaluer la qualité initiale des images (format, taille).
	•	1.2 Nettoyage et normalisation des images
	•	Redimensionnement uniforme à 224x224 pixels.
	•	Filtrage Gaussien (sigma et noyau selon recommandations précédentes).
	•	Normalisation des pixels ([0,1]).
	•	1.3 Gestion des images problématiques
	•	Identifier et supprimer des images corrompues ou non représentatives visuellement.
	•	Vérifier visuellement les échantillons.
	•	1.4 Augmentation des données (Data Augmentation)
	•	Rotations, symétries horizontales et verticales.
	•	Légers zooms et translations.
	•	Validation visuelle rapide.

⸻

✅ Partie 2 : Exploration des données (EDA)
	•	2.1 Analyse descriptive des datasets
	•	Nombre d’images par espèce/maladie.
	•	Identification de déséquilibres (classes majoritaires/minoritaires).
	•	2.2 Exploration statistique rapide des caractéristiques extraites
	•	Calculer moyennes, médianes, variances des caractéristiques initiales (forme, couleur, texture).
	•	Identifier rapidement les éventuels NA (valeurs manquantes).
	•	2.3 Traitement des valeurs manquantes (NA) et aberrantes (Outliers)
	•	Imputation simple par médiane des NA.
	•	Détection des outliers (Isolation Forest recommandé).
	•	Validation rapide par statistiques descriptives post-nettoyage.

⸻

✅ Partie 3 : Data Visualisation (EDA visuelle)
	•	3.1 Visualisation des distributions de classes
	•	Histogrammes / barplots du nombre d’images par classe.
	•	3.2 Visualisation de caractéristiques clés
	•	Boxplots par espèce/maladie (Forme, Texture, Couleur).
	•	Identification visuelle rapide d’outliers via boxplot.
	•	3.3 Réduction de dimensionnalité (PCA / t-SNE) (optionnel mais recommandé)
	•	PCA simple pour visualiser rapidement la séparation des espèces et maladies en 2D.
	•	t-SNE si temps disponible pour une meilleure visualisation des groupes.

⸻

✅ Partie 4 : Feature Engineering
	•	4.1 Extraction des caractéristiques initiales (Baseline)
	•	Forme : Moments de Hu, Fourier descriptors, Solidity.
	•	Texture : GLCM (contraste, énergie, homogénéité), Local Binary Patterns (LBP).
	•	Couleur : HSV (moyenne, std, histogrammes).
	•	Venation : Densité contours (Canny).
	•	4.2 Extraction des caractéristiques avancées (amélioration)
	•	Forme : Eccentricité, ratio Aire/Périmètre.
	•	Texture : Filtres de Gabor.
	•	Couleur : Skewness, Kurtosis en HSV/LAB.
	•	Venation : Éventuellement Sobel/Laplacien avancés.
	•	4.3 Sélection des meilleures caractéristiques
	•	Utiliser Random Forest pour sélectionner les caractéristiques les plus importantes.
	•	Vérifier les scores d’importance des features.
	•	4.4 Validation du Feature Engineering
	•	Tester rapidement (10-fold cross-validation) avec SVM ou Random Forest pour comparer performances initiales avant optimisation.

⸻

🚩 Livrables finaux attendus à ce stade du projet :

✔️ 1. Un rapport d’EDA (Exploratory Data Analysis) clair avec des visualisations :
	•	Statistiques descriptives des datasets.
	•	Visualisations (boxplots, PCA, distributions).

✔️ 2. Un pipeline clair et réutilisable de prétraitement des données :
	•	Code documenté pour traitement NA et outliers.
	•	Code pour extraction complète et standardisée des features.

✔️ 3. Une matrice finale des caractéristiques extraites prête à être utilisée en modèle ML.