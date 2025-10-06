import pandas as pd
import os
from tabulate import tabulate

def get_best_scores(filepath, model_name):
    """Récupère les meilleurs scores pour un modèle donné"""
    df = pd.read_csv(filepath)
    model_df = df[df['Pipeline'] == model_name].copy()
    
    if model_df.empty:
        return None
    
    # Trouver la meilleure classe par F1-score
    best_f1 = model_df.loc[model_df['F1_score'].idxmax()]
    
    return {
        'Modèle': model_name,
        'Classe': best_f1['Classe'],
        'F1-score': best_f1['F1_score'] * 100,
        'Précision': best_f1['Precision'] * 100,
        'Rappel': best_f1['Recall'] * 100,
        'Support': int(best_f1['Support'])
    }

def main():
    # Chemins des fichiers
    base_dir = '/workspaces/datasciencetest_reco_plante/results/models/xgboost'
    maladie_path = os.path.join(base_dir, 'nom_maladie/class_results.csv')
    espece_path = os.path.join(base_dir, 'nom_plante/class_results.csv')
    
    # Modèles à analyser
    models = ['XGBoost', 'XGBoost + PCA', 'XGBoost + LDA']
    
    # Récupérer les meilleurs scores pour chaque modèle
    results = []
    
    # Pour les maladies
    for model in models:
        scores = get_best_scores(maladie_path, model)
        if scores:
            scores['Type'] = 'Maladies'
            results.append(scores)
    
    # Pour les espèces
    for model in models:
        scores = get_best_scores(espece_path, model)
        if scores:
            scores['Type'] = 'Espèces'
            results.append(scores)
    
    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    
    # Réorganiser les colonnes
    df_results = df_results[[
        'Type', 'Modèle', 'Classe', 'F1-score', 'Précision', 'Rappel', 'Support'
    ]]
    
    # Trier par Type puis par F1-score décroissant
    df_results = df_results.sort_values(['Type', 'F1-score'], ascending=[True, False])
    
    # Formater les valeurs numériques
    for col in ['F1-score', 'Précision', 'Rappel']:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.2f}%")
    
    # Générer le tableau Markdown
    table = tabulate(
        df_results,
        headers='keys',
        tablefmt='pipe',
        showindex=False,
        stralign='left',
        numalign='right'
    )
    
    # Ajouter un titre et des commentaires
    markdown = "# Meilleurs scores par modèle\n\n"
    markdown += "Ce tableau présente les meilleures performances obtenues par chaque modèle, pour chaque type de classification.\n\n"
    markdown += table
    
    # Enregistrer dans un fichier
    output_path = os.path.join(base_dir, 'best_scores_summary.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"Résumé des meilleurs scores sauvegardé : {output_path}")
    print("\nRésumé des meilleurs scores :")
    print("-" * 100)
    print(markdown)
    print("-" * 100)

if __name__ == "__main__":
    main()
