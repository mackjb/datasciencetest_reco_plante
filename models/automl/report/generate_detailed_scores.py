import pandas as pd
from tabulate import tabulate
import os

def load_and_prepare_data(filepath, model_type):
    """Charge et prépare les données pour un type de modèle spécifique"""
    df = pd.read_csv(filepath)
    
    # Filtrer pour le type de modèle spécifié
    if model_type == 'XGBoost':
        df = df[df['Pipeline'] == 'XGBoost']
    elif model_type == 'XGBoost + PCA':
        df = df[df['Pipeline'] == 'XGBoost + PCA']
    elif model_type == 'XGBoost + LDA':
        df = df[df['Pipeline'] == 'XGBoost + LDA']
    
    # Supprimer les doublons en gardant la première occurrence
    df = df.drop_duplicates(subset=['Classe'], keep='first')
    
    # Sélectionner les colonnes pertinentes
    df = df[['Classe', 'Precision', 'Recall', 'F1_score', 'Support']]
    
    # Trier par F1_score décroissant
    return df.sort_values('F1_score', ascending=False)

def generate_model_report(df, model_name, output_dir):
    """Génère un rapport détaillé pour un modèle spécifique"""
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculer les moyennes
    avg_precision = df['Precision'].mean() * 100
    avg_recall = df['Recall'].mean() * 100
    avg_f1 = df['F1_score'].mean() * 100
    
    # Formater les données pour l'affichage
    df_display = df.copy()
    df_display['Precision'] = (df_display['Precision'] * 100).round(2).astype(str) + '%'
    df_display['Recall'] = (df_display['Recall'] * 100).round(2).astype(str) + '%'
    df_display['F1_score'] = (df_display['F1_score'] * 100).round(2).astype(str) + '%'
    
    # Générer le contenu du rapport
    report = f"# Rapport détaillé - {model_name}\n\n"
    report += f"- **Nombre de classes**: {len(df)}\n"
    report += f"- **Précision moyenne**: {avg_precision:.2f}%\n"
    report += f"- **Rappel moyen**: {avg_recall:.2f}%\n"
    report += f"- **Score F1 moyen**: {avg_f1:.2f}%\n\n"
    
    # Ajouter le tableau des performances par classe
    report += "## Performances par classe\n\n"
    report += tabulate(
        df_display,
        headers=['Classe', 'Précision', 'Rappel', 'F1-score', 'Support'],
        tablefmt='pipe',
        showindex=False
    )
    
    # Sauvegarder le rapport
    output_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Rapport généré: {output_path}")
    return report

def main():
    # Chemins des fichiers
    base_dir = '/workspaces/datasciencetest_reco_plante/results/models/xgboost'
    maladie_path = os.path.join(base_dir, 'nom_maladie/class_results.csv')
    espece_path = os.path.join(base_dir, 'nom_plante/class_results.csv')
    
    # Modèles à analyser
    models = ['XGBoost', 'XGBoost + PCA', 'XGBoost + LDA']
    
    # Générer les rapports pour chaque modèle et chaque type de données
    for model in models:
        print(f"\n=== {model} ===")
        
        # Maladies
        print("\nAnalyse des maladies...")
        df_maladie = load_and_prepare_data(maladie_path, model)
        generate_model_report(
            df_maladie, 
            f"{model} - Maladies",
            os.path.join(base_dir, 'reports')
        )
        
        # Espèces
        print("Analyse des espèces...")
        df_espece = load_and_prepare_data(espece_path, model)
        generate_model_report(
            df_espece,
            f"{model} - Espèces",
            os.path.join(base_dir, 'reports')
        )
    
    print("\nTous les rapports ont été générés avec succès!")

if __name__ == "__main__":
    main()
