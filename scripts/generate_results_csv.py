#!/usr/bin/env python3
# Script pour générer un CSV des résultats des modèles

import pandas as pd

# Données des résultats
results = [
    # Modèles pour la classification d'espèces
    {
        'Tâche': 'Espèces',
        'Modèle': 'Light Gradient Boosting Machine',
        'Accuracy': 0.9003,
        'F1-Score': 0.8997,
        'AUC': 0.9929,
        'Recall': 0.9003,
        'Precision': 0.8998,
        'Meilleur': 'Oui'
    },
    {
        'Tâche': 'Espèces',
        'Modèle': 'Random Forest Classifier',
        'Accuracy': 0.8725,
        'F1-Score': 0.8718,
        'AUC': 0.9888,
        'Recall': 0.8725,
        'Precision': 0.8730,
        'Meilleur': 'Non'
    },
    {
        'Tâche': 'Espèces',
        'Modèle': 'Extra Trees Classifier',
        'Accuracy': 0.8700,
        'F1-Score': 0.8687,
        'AUC': 0.9881,
        'Recall': 0.8700,
        'Precision': 0.8698,
        'Meilleur': 'Non'
    },
    {
        'Tâche': 'Espèces',
        'Modèle': 'K Neighbors Classifier',
        'Accuracy': 0.7615,
        'F1-Score': 0.7680,
        'AUC': 0.9341,
        'Recall': 0.7615,
        'Precision': 0.7922,
        'Meilleur': 'Non'
    },
    {
        'Tâche': 'Espèces',
        'Modèle': 'Linear Discriminant Analysis',
        'Accuracy': 0.7355,
        'F1-Score': 0.7421,
        'AUC': 0.0000,
        'Recall': 0.7355,
        'Precision': 0.7694,
        'Meilleur': 'Non'
    },
    # Modèles pour la classification de maladies
    {
        'Tâche': 'Maladies',
        'Modèle': 'Light Gradient Boosting Machine',
        'Accuracy': 0.8142,
        'F1-Score': 0.8148,
        'AUC': 0.9867,
        'Recall': 0.8142,
        'Precision': 0.8160,
        'Meilleur': 'Oui'
    },
    {
        'Tâche': 'Maladies',
        'Modèle': 'Random Forest Classifier',
        'Accuracy': 0.7759,
        'F1-Score': 0.7773,
        'AUC': 0.9796,
        'Recall': 0.7759,
        'Precision': 0.7825,
        'Meilleur': 'Non'
    },
    {
        'Tâche': 'Maladies',
        'Modèle': 'Extra Trees Classifier',
        'Accuracy': 0.7734,
        'F1-Score': 0.7743,
        'AUC': 0.9781,
        'Recall': 0.7734,
        'Precision': 0.7798,
        'Meilleur': 'Non'
    },
    {
        'Tâche': 'Maladies',
        'Modèle': 'K Neighbors Classifier',
        'Accuracy': 0.6354,
        'F1-Score': 0.6489,
        'AUC': 0.8952,
        'Recall': 0.6354,
        'Precision': 0.6819,
        'Meilleur': 'Non'
    },
    {
        'Tâche': 'Maladies',
        'Modèle': 'Linear Discriminant Analysis',
        'Accuracy': 0.6130,
        'F1-Score': 0.6252,
        'AUC': 0.0000,
        'Recall': 0.6130,
        'Precision': 0.6775,
        'Meilleur': 'Non'
    }
]

def main():
    # Créer un DataFrame
    df = pd.DataFrame(results)
    
    # Trier par tâche et par F1-Score (décroissant)
    df = df.sort_values(by=['Tâche', 'F1-Score'], ascending=[True, False])
    
    # Formater les nombres pour un affichage plus lisible
    for col in ['Accuracy', 'F1-Score', 'AUC', 'Recall', 'Precision']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    # Enregistrer dans un fichier CSV
    output_file = 'results/comparaison_modeles.csv'
    df.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"Fichier CSV généré avec succès : {output_file}")
    
    # Afficher un aperçu du DataFrame
    print("\nAperçu des données :")
    print(df.head(10))

if __name__ == "__main__":
    main()
