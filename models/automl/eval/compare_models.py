import pandas as pd
from tabulate import tabulate
import os

def load_data(filepath):
    """Charge les données à partir du fichier CSV"""
    return pd.read_csv(filepath)

def get_model_performance(df, model_name):
    """Calcule les performances moyennes pour un modèle donné"""
    model_df = df[df['Pipeline'] == model_name].copy()
    
    if model_df.empty:
        return None
    
    # Calculer les moyennes globales
    performance = {
        'Modèle': model_name,
        'F1_moyen': model_df['F1_score'].mean() * 100,
        'Précision_moyenne': model_df['Precision'].mean() * 100,
        'Rappel_moyen': model_df['Recall'].mean() * 100,
        'Nb_classes': model_df['Classe'].nunique()
    }
    
    return performance

def compare_models(maladie_path, espece_path):
    """Compare les performances de tous les modèles"""
    # Charger les données
    maladie_df = load_data(maladie_path)
    espece_df = load_data(espece_path)
    
    # Identifier tous les modèles uniques
    all_models = list(set(maladie_df['Pipeline'].unique()).union(set(espece_df['Pipeline'].unique())))
    
    # Calculer les performances pour chaque modèle
    results = []
    
    for model in all_models:
        # Performances sur les maladies
        perf_maladie = get_model_performance(maladie_df, model)
        if perf_maladie:
            perf_maladie['Type'] = 'Maladies'
            results.append(perf_maladie)
        
        # Performances sur les espèces
        perf_espece = get_model_performance(espece_df, model)
        if perf_espece:
            perf_espece['Type'] = 'Espèces'
            results.append(perf_espece)
    
    # Créer un DataFrame avec les résultats
    comparison_df = pd.DataFrame(results)
    
    # Réorganiser les colonnes
    comparison_df = comparison_df[['Modèle', 'Type', 'Nb_classes', 'F1_moyen', 'Précision_moyenne', 'Rappel_moyen']]
    
    # Trier par Type puis par F1_moyen décroissant
    comparison_df = comparison_df.sort_values(['Type', 'F1_moyen'], ascending=[True, False])
    
    return comparison_df

def generate_report(comparison_df):
    """Génère un rapport de comparaison des modèles"""
    output = "# Comparaison des modèles de classification\n\n"
    
    # Tableau comparatif
    table = tabulate(
        comparison_df,
        headers=['Modèle', 'Type', 'Nb classes', 'F1 moyen', 'Précision', 'Rappel'],
        tablefmt='pipe',
        showindex=False,
        floatfmt='.2f',
        numalign='right',
        stralign='left'
    )
    
    output += table + "\n\n"
    
    # Meilleurs modèles par catégorie
    for model_type in ['Maladies', 'Espèces']:
        subset = comparison_df[comparison_df['Type'] == model_type]
        if not subset.empty:
            best = subset.iloc[0]
            output += f"## Meilleur modèle pour les {model_type}: {best['Modèle']}\n"
            output += f"- Score F1: {best['F1_moyen']:.2f}%\n"
            output += f"- Précision: {best['Précision_moyenne']:.2f}%\n"
            output += f"- Rappel: {best['Rappel_moyen']:.2f}%\n\n"
    
    # Sauvegarder le rapport
    output_path = '/workspaces/datasciencetest_reco_plante/results/model_comparison.md'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"Rapport de comparaison généré: {output_path}")
    print("\nRésumé des performances par modèle:")
    print("-" * 100)
    print(output)
    print("-" * 100)

def main():
    # Chemins des fichiers
    maladie_path = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_maladie/class_results.csv'
    espece_path = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_plante/class_results.csv'
    
    # Comparer les modèles
    comparison_df = compare_models(maladie_path, espece_path)
    
    # Générer le rapport
    generate_report(comparison_df)

if __name__ == "__main__":
    main()
