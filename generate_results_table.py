import pandas as pd
import numpy as np
from tabulate import tabulate
import os

def load_and_prepare_data(filepath):
    """Charge et prépare les données à partir du fichier CSV"""
    df = pd.read_csv(filepath)
    
    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Extraire la configuration de base (sans PCA/LDA)
    df_base = df[df['Pipeline'] == 'XGBoost'].copy()
    
    # Calculer les moyennes par classe
    class_means = df_base.groupby('Classe').agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1_score': 'mean',
        'Support': 'first'  # Le support est le même pour chaque classe
    }).reset_index()
    
    return class_means.sort_values('F1_score', ascending=False)

def create_formatted_table(df, title, top_n=15):
    """Crée un tableau formaté à partir du DataFrame"""
    # Sélectionner les top_n classes
    top_df = df.head(top_n).copy()
    
    # Formater les colonnes numériques
    for col in ['Precision', 'Recall', 'F1_score']:
        top_df[col] = top_df[col].apply(lambda x: f"{x:.2%}")
    
    # Renommer les colonnes pour l'affichage
    top_df = top_df.rename(columns={
        'Classe': 'Classe',
        'Precision': 'Précision',
        'Recall': 'Rappel',
        'F1_score': 'F1-score',
        'Support': 'Nb. échantillons'
    })
    
    # Créer le tableau formaté
    table = tabulate(
        top_df,
        headers='keys',
        tablefmt='pipe',
        showindex=False,
        numalign='right',
        stralign='left',
        floatfmt='.2f'
    )
    
    return f"\n### {title}\n\n{table}\n"

def main():
    # Chemins des fichiers
    maladie_path = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_maladie/class_results.csv'
    espece_path = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_plante/class_results.csv'
    
    # Charger et traiter les données
    maladie_df = load_and_prepare_data(maladie_path)
    espece_df = load_and_prepare_data(espece_path)
    
    # Créer les tableaux formatés
    output = "# Résultats de classification\n\n"
    
    # Tableau des maladies
    output += create_formatted_table(
        maladie_df, 
        "Top 15 des maladies les mieux classées (par F1-score)",
        top_n=15
    )
    
    # Tableau des espèces
    output += create_formatted_table(
        espece_df,
        "Top 15 des espèces les mieux classées (par F1-score)",
        top_n=15
    )
    
    # Ajouter des statistiques globales
    output += "## Statistiques globales\n\n"
    
    # Pour les maladies
    output += "### Maladies\n"
    output += f"- Nombre total de classes: {len(maladie_df)}\n"
    output += f"- Score F1 moyen: {maladie_df['F1_score'].mean():.2%}\n"
    output += f"- Précision moyenne: {maladie_df['Precision'].mean():.2%}\n"
    output += f"- Rappel moyen: {maladie_df['Recall'].mean():.2%}\n\n"
    
    # Pour les espèces
    output += "### Espèces\n"
    output += f"- Nombre total de classes: {len(espece_df)}\n"
    output += f"- Score F1 moyen: {espece_df['F1_score'].mean():.2%}\n"
    output += f"- Précision moyenne: {espece_df['Precision'].mean():.2%}\n"
    output += f"- Rappel moyen: {espece_df['Recall'].mean():.2%}\n"
    
    # Sauvegarder dans un fichier Markdown
    output_path = '/workspaces/datasciencetest_reco_plante/results/classification_results.md'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"Rapport généré avec succès: {output_path}")
    print("\nPour visualiser les résultats, ouvrez le fichier Markdown dans VS Code ou consultez son contenu ci-dessous:")
    print("\n" + "="*80)
    print(output)
    print("="*80)

if __name__ == "__main__":
    main()
