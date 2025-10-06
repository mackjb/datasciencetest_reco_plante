import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results():
    """Charge les résultats des modèles"""
    base_dir = 'results/models/xgboost'
    
    # Charger les résultats pour les maladies et les espèces
    maladies = pd.read_csv(f'{base_dir}/nom_maladie/class_results.csv')
    especes = pd.read_csv(f'{base_dir}/nom_plante/class_results.csv')
    
    # Ajouter une colonne pour le type de données
    maladies['Type'] = 'Maladies'
    especes['Type'] = 'Espèces'
    
    # Concaténer les résultats
    return pd.concat([maladies, especes], ignore_index=True)

def plot_side_by_side_comparison(results_df):
    """Affiche une comparaison côte à côte des métriques"""
    metrics = ['Precision', 'Recall', 'F1_score']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        # Préparer les données avec écart-type
        plot_data = results_df.groupby(['Type', 'Pipeline'])[metric].agg(['mean', 'std']).unstack()
        
        # Tracer avec barres d'erreur
        plot_data['mean'].plot(kind='bar', ax=axes[i], yerr=plot_data['std'], 
                             capsize=4, ecolor='black', alpha=0.8)
        
        axes[i].set_title(f'{metric} par modèle')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title='Modèle', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures/side_by_side_comparison.png', bbox_inches='tight')
    plt.close()

def plot_per_class_metrics(results_df, metric='F1_score'):
    """Affiche les métriques par classe pour chaque modèle"""
    for data_type in results_df['Type'].unique():
        plt.figure(figsize=(15, 8))
        
        # Filtrer les données
        df = results_df[results_df['Type'] == data_type]
        
        # Créer un boxplot par modèle
        df.boxplot(column=metric, by=['Pipeline', 'Classe'], 
                  grid=False, rot=90, fontsize=8)
        
        plt.title(f'Distribution du {metric} par classe - {data_type}')
        plt.suptitle('')
        plt.xlabel('Modèle et Classe')
        plt.ylabel(metric)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig(f'figures/per_class_{metric.lower()}_{data_type.lower()}.png', 
                   bbox_inches='tight')
        plt.close()

def analyze_overfitting(results_df):
    """Analyse le surapprentissage à partir des résultats"""
    # Calculer les moyennes par modèle et type
    metrics = ['Precision', 'Recall', 'F1_score']
    
    # Préparer les résultats
    analysis = {}
    
    for (model_type, pipeline), group in results_df.groupby(['Type', 'Pipeline']):
        if pipeline not in analysis:
            analysis[pipeline] = {}
        
        # Calculer les métriques moyennes et écart-type
        analysis[pipeline][model_type] = {
            'mean': {
                metric: group[metric].mean() * 100 
                for metric in metrics
            },
            'std': {
                metric: group[metric].std() * 100
                for metric in metrics
            },
            'min': {
                metric: group[metric].min() * 100
                for metric in metrics
            },
            'max': {
                metric: group[metric].max() * 100
                for metric in metrics
            },
            'n_classes': group['Classe'].nunique()
        }
    
    return analysis

def print_analysis(analysis):
    """Affiche l'analyse du surapprentissage"""
    print("\n" + "="*100)
    print("ANALYSE DÉTAILLÉE DES PERFORMANCES ET DU SURAPPRENTISSAGE")
    print("="*100)
    
    for pipeline, data in analysis.items():
        print(f"\n\n{'='*50}")
        print(f"MODÈLE: {pipeline}")
        print(f"{'='*50}")
        
        for model_type, metrics in data.items():
            print(f"\n🔍 {model_type.upper()}:")
            print(f"   {'Métrique':<15} {'Moyenne':<10} {'Écart-type':<12} {'Min':<10} {'Max':<10}")
            print(f"   {'-'*15} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
            
            for metric in ['Precision', 'Recall', 'F1_score']:
                mean = metrics['mean'][metric]
                std = metrics['std'][metric]
                min_val = metrics['min'][metric]
                max_val = metrics['max'][metric]
                
                # Déterminer l'icône en fonction de la variance
                if std > 15:
                    icon = "⚠️"
                elif std < 5:
                    icon = "✅"
                else:
                    icon = "ℹ️"
                
                print(f"   {icon} {metric:<12} {mean:>6.2f}% ±{std:>5.2f}%  {min_val:>6.2f}%  {max_val:>6.2f}%")
            
            # Afficher l'interprétation
            print("\n   INTERPRÉTATION:")
            if metrics['std']['F1_score'] > 15:
                print("   ⚠️  Forte variance entre les classes - Risque de surapprentissage sur certaines classes")
            elif metrics['std']['F1_score'] < 5:
                print("   ✅  Faible variance - Bonne cohérence entre les classes")
            else:
                print("   ℹ️  Variance modérée - Certaines classes peuvent nécessiter plus d'attention")
            
            # Vérifier l'écart min-max
            f1_range = metrics['max']['F1_score'] - metrics['min']['F1_score']
            if f1_range > 40:
                print(f"   ⚠️  Grand écart de performance entre classes ({f1_range:.1f}%) - Vérifier les classes problématiques")
            
            print(f"\n   Nombre de classes: {metrics['n_classes']}")

def main():
    print("Chargement et préparation des résultats...")
    results_df = load_results()
    
    # Nettoyer les données
    results_df = results_df[results_df['Config'] == 'Baseline']
    
    print("\nAnalyse des performances...")
    analysis = analyze_overfitting(results_df)
    
    # Créer le dossier pour les figures
    os.makedirs('figures', exist_ok=True)
    
    print("\nGénération des visualisations...")
    # Graphique comparatif côte à côte
    plot_side_by_side_comparison(results_df)
    
    # Graphiques par classe
    for metric in ['F1_score', 'Precision', 'Recall']:
        plot_per_class_metrics(results_df, metric)
    
    # Afficher l'analyse détaillée
    print("\n" + "="*100)
    print("RÉSULTATS DÉTAILLÉS PAR MODÈLE")
    print("="*100)
    print_analysis(analysis)
    
    print("\n" + "="*100)
    print("ANALYSE TERMINÉE")
    print("="*100)
    print("Les graphiques ont été sauvegardés dans le dossier 'figures/'")
    print("\nRÉCAPITULATIF DES PRINCIPAUX POINTS :")
    print("-" * 50)
    print("1. Comparaison des modèles : Voir 'figures/side_by_side_comparison.png'")
    print("2. Détail par classe : Voir les fichiers 'per_class_*_*.png'")
    print("3. Analyse complète : Voir le rapport ci-dessus pour les détails par modèle")

if __name__ == "__main__":
    main()
