import pandas as pd
import os
from tabulate import tabulate

def calculate_average_scores(filepath, model_name):
    """Calcule les moyennes pour un modèle donné"""
    df = pd.read_csv(filepath)
    model_df = df[df['Pipeline'] == model_name].copy()
    
    if model_df.empty:
        return None
    
    return {
        'Modèle': model_name,
        'F1_moyen': model_df['F1_score'].mean() * 100,
        'Précision_moyenne': model_df['Precision'].mean() * 100,
        'Rappel_moyen': model_df['Recall'].mean() * 100,
        'Nb_classes': model_df['Classe'].nunique()
    }

def main():
    # Chemins des fichiers
    base_dir = '/workspaces/datasciencetest_reco_plante/results/models/xgboost'
    maladie_path = os.path.join(base_dir, 'nom_maladie/class_results.csv')
    espece_path = os.path.join(base_dir, 'nom_plante/class_results.csv')
    
    # Modèles à analyser
    models = ['XGBoost', 'XGBoost + PCA', 'XGBoost + LDA']
    
    # Calculer les moyennes pour chaque modèle
    results = []
    
    # Pour les maladies
    for model in models:
        scores = calculate_average_scores(maladie_path, model)
        if scores:
            scores['Type'] = 'Maladies'
            results.append(scores)
    
    # Pour les espèces
    for model in models:
        scores = calculate_average_scores(espece_path, model)
        if scores:
            scores['Type'] = 'Espèces'
            results.append(scores)
    
    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    
    # Réorganiser les colonnes
    df_results = df_results[[
        'Type', 'Modèle', 'Nb_classes', 'F1_moyen', 'Précision_moyenne', 'Rappel_moyen'
    ]]
    
    # Trier par Type puis par F1_moyen décroissant
    df_results = df_results.sort_values(['Type', 'F1_moyen'], ascending=[True, False])
    
    # Formater les valeurs numériques
    for col in ['F1_moyen', 'Précision_moyenne', 'Rappel_moyen']:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.2f}%")
    
    # Générer le tableau Markdown
    table = tabulate(
        df_results,
        headers=['Type', 'Modèle', 'Nb classes', 'F1 moyen', 'Précision', 'Rappel'],
        tablefmt='pipe',
        showindex=False,
        stralign='left',
        numalign='right'
    )
    
    # Ajouter un titre
    markdown = "# Comparaison des performances moyennes par modèle\n\n"
    markdown += table
    
    # Enregistrer dans un fichier
    output_path = os.path.join(base_dir, 'model_comparison_summary.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"Résumé des performances sauvegardé : {output_path}")
    print("\nComparaison des modèles :")
    print("-" * 100)
    print(markdown)
    print("-" * 100)

if __name__ == "__main__":
    main()
