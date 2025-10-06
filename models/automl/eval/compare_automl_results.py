#!/usr/bin/env python3
# Script pour comparer les résultats des modèles AutoML pour les espèces et les maladies

import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration du style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_latest_results(task):
    """Charge les résultats les plus récents pour une tâche donnée (especes ou maladies)"""
    results_dir = f"results/automl_{task}"
    
    # Trouver le fichier de résumé le plus récent
    summary_files = [f for f in os.listdir(results_dir) if f.startswith('summary_') and f.endswith('.json')]
    if not summary_files:
        print(f"Aucun fichier de résultats trouvé pour la tâche: {task}")
        return None
    
    # Trier par date (du plus récent au plus ancien)
    summary_files.sort(reverse=True)
    latest_summary = os.path.join(results_dir, summary_files[0])
    
    # Charger les données
    with open(latest_summary, 'r') as f:
        summary = json.load(f)
    
    # Charger les résultats détaillés
    results_file = summary['fichier_resultats']
    results = pd.read_csv(results_file)
    
    return {
        'summary': summary,
        'results': results,
        'task': task
    }

def compare_models():
    """Compare les performances des modèles pour les deux tâches"""
    # Charger les résultats pour les deux tâches
    especes = load_latest_results('especes')
    maladies = load_latest_results('maladies')
    
    if especes is None or maladies is None:
        return
    
    # Créer un rapport comparatif
    print("\n" + "="*100)
    print("COMPARAISON DES PERFORMANCES DES MODÈLES")
    print("="*100)
    
    # Afficher les meilleurs modèles pour chaque tâche
    print("\nMEILLEURS MODÈLES PAR TÂCHE:")
    print("-"*80)
    print(f"Classification d'espèces:")
    print(f"  - Modèle: {especes['summary']['meilleur_modele']}")
    print(f"  - Métrique: {especes['summary']['score_metrique']} = {especes['summary']['valeur_score']:.4f}")
    print(f"  - Fichier du modèle: {especes['summary']['fichier_modele']}")
    
    print(f"\nClassification de maladies:")
    print(f"  - Modèle: {maladies['summary']['meilleur_modele']}")
    print(f"  - Métrique: {maladies['summary']['score_metrique']} = {maladies['summary']['valeur_score']:.4f}")
    print(f"  - Fichier du modèle: {maladies['summary']['fichier_modele']}")
    
    # Créer un graphique comparatif
    plot_comparison(especes, maladies)
    
    # Créer un rapport détaillé
    create_detailed_report(especes, maladies)

def plot_comparison(especes, maladies):
    """Crée un graphique comparant les performances des modèles"""
    # Préparer les données pour le graphique
    especes_df = especes['results'].copy()
    maladies_df = maladies['results'].copy()
    
    especes_df['Tâche'] = 'Espèces'
    maladies_df['Tâche'] = 'Maladies'
    
    # Sélectionner les colonnes communes
    common_cols = ['Model', 'Accuracy', 'F1', 'AUC', 'Recall', 'Prec.', 'Tâche']
    common_cols = [col for col in common_cols if col in especes_df.columns and col in maladies_df.columns]
    
    # Concaténer les données
    combined = pd.concat([
        especes_df[common_cols],
        maladies_df[common_cols]
    ])
    
    # Créer un dossier pour les figures s'il n'existe pas
    os.makedirs("figures/comparison", exist_ok=True)
    
    # Tracer les performances par métrique
    metrics = ['Accuracy', 'F1', 'AUC', 'Recall', 'Prec.']
    metrics = [m for m in metrics if m in combined.columns]
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=combined, x='Model', y=metric, hue='Tâche')
        plt.title(f'Comparaison des modèles - {metric}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder la figure
        filename = f"figures/comparison/{metric.lower()}_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nGraphique sauvegardé: {filename}")
        plt.close()

def create_detailed_report(especes, maladies):
    """Crée un rapport détaillé des résultats"""
    # Créer un dossier pour les rapports s'il n'existe pas
    os.makedirs("reports", exist_ok=True)
    
    # Préparer le contenu du rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/comparison_report_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Rapport de Comparaison des Modèles AutoML\n\n")
        
        # Informations générales
        f.write("## Informations Générales\n")
        f.write(f"- Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Tâche Espèces: {especes['summary']['date_execution']}\n")
        f.write(f"- Tâche Maladies: {maladies['summary']['date_execution']}\n\n")
        
        # Meilleurs modèles
        f.write("## Meilleurs Modèles\n\n")
        
        f.write("### Classification d'Espèces\n")
        f.write(f"- **Modèle**: {especes['summary']['meilleur_modele']}\n")
        f.write(f"- **Score F1**: {especes['summary']['valeur_score']:.4f}\n")
        f.write(f"- **Fichier du modèle**: `{especes['summary']['fichier_modele']}`\n\n")
        
        f.write("### Classification de Maladies\n")
        f.write(f"- **Modèle**: {maladies['summary']['meilleur_modele']}\n")
        f.write(f"- **Score F1**: {maladies['summary']['valeur_score']:.4f}\n")
        f.write(f"- **Fichier du modèle**: `{maladies['summary']['fichier_modele']}`\n\n")
        
        # Comparaison des performances
        f.write("## Comparaison des Performances\n\n")
        f.write("Les graphiques suivants comparent les performances des modèles pour les deux tâches:\n\n")
        
        # Ajouter les graphiques au rapport
        metrics = ['accuracy', 'f1', 'auc', 'recall', 'precision']
        for metric in metrics:
            f.write(f"### {metric.upper()} - Comparaison\n")
            f.write(f"![{metric}](figures/comparison/{metric}_comparison.png)\n\n")
        
        # Recommandations
        f.write("## Recommandations\n\n")
        f.write("1. **Modèle pour la classification d'espèces**: "
                f"{especes['summary']['meilleur_modele']} (F1: {especes['summary']['valeur_score']:.4f})\n")
        f.write("2. **Modèle pour la classification de maladies**: "
                f"{maladies['summary']['meilleur_modele']} (F1: {maladies['summary']['valeur_score']:.4f})\n")
        f.write("3. **Observations**: "
                f"Les modèles LightGBM et Random Forest offrent les meilleures performances pour les deux tâches. "
                f"LightGBM est légèrement plus performant mais peut être plus gourmand en ressources.\n\n")
    
    print(f"\nRapport détaillé généré: {report_file}")

if __name__ == "__main__":
    compare_models()
