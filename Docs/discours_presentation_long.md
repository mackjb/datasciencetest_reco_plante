# Discours de Présentation Détaillé : Méthodologie & Machine Learning
**Durée estimée : ~6 minutes**

---

## 1. Introduction & Stratégie (Page "Méthodologie")
*(~1 minute)*

"Bonjour à tous. J'aimerais commencer cette présentation par la page **Méthodologie**, qui pose les fondations de notre approche.

Comme vous le savez, la littérature récente place le Deep Learning comme la solution reine pour la reconnaissance d'images. Cependant, nous avons fait le choix délibéré de ne pas nous précipiter uniquement sur des réseaux de neurones complexes.

Notre stratégie repose sur une **comparaison rigoureuse** :
1.  **Une Baseline Machine Learning (à gauche)** : Nous avons voulu construire un modèle "expert", où chaque étape est maîtrisée et chaque descripteur est choisi. Cela nous donne une référence interprétable et un seuil de performance à battre.
2.  **Une Architecture Deep Learning (à droite)** : Qui sera déployée dans un second temps pour repousser les limites de la performance.

Cette approche comparative est cruciale : elle nous permet de dire précisément *ce que le Deep Learning apporte de plus* que de simples descripteurs de forme ou de couleur."

---

## 2. Le Pipeline Machine Learning (Page "Machine Learning")
*(~4 minutes 30)*

"Je vous invite maintenant à explorer l'onglet **Machine Learning**. Pour rendre ce processus technique plus digeste, nous avons modélisé notre pipeline en **6 étapes chronologiques**. Vous pouvez naviguer avec moi en cliquant sur les boutons."

*(Cliquez sur **1. Extraction**)*
"**Étape 1 : L'Extraction de Features.**
Le défi du Machine Learning classique, c'est que les algorithmes ne "voient" pas l'image. Nous devons la traduire en chiffres.
Nous avons extrait **34 descripteurs** mathématiques pour chaque feuille :
*   **La Forme** : Aire, périmètre, mais aussi les *Moments de Hu* qui sont très puissants car ils restent identiques même si la feuille est tournée ou zoomée.
*   **La Texture** : Via les matrices de Haralick (contraste, homogénéité), essentielles pour repérer les taches de maladies rugueuses.
*   **La Couleur** : Moyennes RGB et HSV, pour détecter le jaunissement ou les nécroses.
C'est cette richesse d'information (le tableau que vous voyez ici) qui nourrit nos modèles."

*(Cliquez sur **2. Split**)*
"**Étape 2 : Le Split Stratifié.**
Pas de surprise ici, mais de la rigueur. Nous avons divisé nos données en 80/10/10.
Le point clé est la **stratification** : nous nous sommes assurés que chaque espèce soit représentée de manière identique dans le Train et le Test. C'est la seule façon d'éviter qu'une espèce rare ne disparaisse complètement du jeu de test."

*(Cliquez sur **3. Rééchantillonnage**)*
"**Étape 3 : Gestion du Déséquilibre.**
Nos données brutes étaient biaisées : la Tomate était sur-représentée.
Sans correction, le modèle aurait appris à toujours prédire "Tomate" pour maximiser son score.
Nous avons donc appliqué des techniques de **SMOTE (Oversampling)** pour créer des exemples synthétiques des classes minoritaires, et de l'Undersampling pour réduire les classes majoritaires. Cela rééquilibre le terrain de jeu."

*(Cliquez sur **4. Pré-traitements**)*
"**Étape 4 : Les Pré-traitements.**
C'est souvent l'étape invisible qui fait gagner le plus de points.
*   **Data Augmentation** : Nous avons artificiellement généré de nouvelles images (rotations, miroirs, bruit) pour passer de 43 000 à près de **92 000 images** d'entraînement. Cela force le modèle à être plus robuste.
*   **Scaling** : Un point crucial ici. Nous avons choisi le **RobustScaler**. Pourquoi ? Parce que 40% de nos features contenaient des **outliers**. Un scaler classique (StandardScaler) aurait été écrasé par ces valeurs extrêmes. Le RobustScaler, basé sur la médiane, les ignore.
*   **Sélection** : Enfin, nous avons utilisé l'analyse **SHAP** pour ne garder que les features qui aident vraiment à la décision, éliminant le bruit."

*(Cliquez sur **5. Modélisation**)*
"**Étape 5 : La Compétition.**
Nous avons lancé 4 modèles en parallèle.
*   Le **SVM** pour sa capacité à trouver des frontières complexes dans des espaces de grande dimension.
*   Le **XGBoost** et les **Extra-Trees** pour la puissance des méthodes ensemblistes.
*   La **Régression Logistique** comme baseline simple."

*(Cliquez sur **6. Évaluation**)*
"**Étape 6 : Les Résultats.**
Regardons le graphique. Le **SVM à noyau RBF** (la barre la plus haute) atteint **93.70% d'accuracy** sur le jeu de test.
C'est un résultat remarquable pour une méthode "classique", sans réseau de neurones profond.

Si on regarde le détail par classe *(montrer le tableau)* :
Le modèle est excellent sur la **Tomate** et le **Blueberry**.
Il a plus de mal sur le *Poivron (Pepper_bell)* ou la *Pomme de terre (Potato)*, souvent à cause de ressemblances morphologiques fortes entre feuilles saines et malades à un stade précoce.
Mais globalement, le F1-score reste supérieur à 90% pour la majorité des espèces."

---

## 3. Conclusion & Transition
*(~30 secondes)*

"Pour conclure sur cette partie :
Nous avons prouvé qu'avec un **Feature Engineering métier** (forme, texture, couleur) et un pipeline rigoureux (RobustScaler, Augmentation), on peut atteindre un niveau de performance quasi-industriel (94%).

Cependant, pour aller chercher les derniers pourcents et gérer des cas plus complexes (fonds bruités, lumières difficiles), nous avons besoin de la capacité d'abstraction du **Deep Learning**. C'est ce que nous allons voir dans la partie suivante."
