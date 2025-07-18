# TODO

JBM : JB
SL : Lionel
MP : Morgan
BG : Bernadette



rajouter la possibilité un générer csv nommé plantvillage_segmented_all.csv
avec une colonne is_na 
is_image_valid 
is_black 
is_duplicate 
is_fail_segmented 



raw_data_plantvillage_segmented_all.csv => repertoire historique 





un nouveau repertoire image avec toutes les images des 3 csv segmented_clean_augmented

quelle résolution image 256*256
quelle format 

On met sous controle de source avec LFS

clean_data_plantvillage_segmented_all.csv
pas de NA et corrige les images en extensions et la même résolution 

augmented_data_plantvillage_segmented_all.csv
flip, scale, rotation, 
=> Bernadette propose de faire des images avec des transformations plus dans les classes minoritaires





Slide pour expliquer l’industrialisation du code 

Integer le standard image corrompue et l’image doublon


Intégrer les features Bernadettes 


Faire 2 CSV clean et outliers

Puis feature importance
Radom forest 
PCA



1bis. Utiliser isblack pour éliminer les 5 images complétement KO, seuil à <5

Intégrer la détection des doublons (images comparées, puis avec feature)


1. Calculer le pixel ratio et mettre en visu les vignettes comme l'a fait morgane pour afficher les images jugées mal semgentée
 
2. sauvegarder et mettre sous controle de source le cacul all sous format cvs du dataframe
 
3. produire un CSV qui liste toutes les images filtrées avc les caractéristiques calculées et le path de l'image modifiée associé, donc un repertoire d'images associées
  
4. appliquer la mécanique sickit learn d'extension du dataset et produire un nouveau csv avec un nouveau repertoire d'images (flip/loop/etc)
 
5. classifier les images avec un modèle de classification (RandomForest) 

6. Etendre aux autres datasets (lesquels?)

6bis. isoler la feature maladie de plant village 

7. retenter une segmentation du dataset plantvillage avec SAM (GPU)




Laplacien
