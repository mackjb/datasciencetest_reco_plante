import pandas as pd
import joblib

# Charger les données de base
df = pd.read_csv('../results/models/xgboost/class_results.csv')

# Charger les résultats d'optimisation
try:
    opt_espece = joblib.load('../results/fast_optimization/results_espece.pkl')
    opt_maladie = joblib.load('../results/fast_optimization/results_maladie.pkl')
except Exception as e:
    print(f"Erreur lors du chargement des fichiers d'optimisation: {e}")

# Fonction pour calculer les moyennes
def get_stats(df, modele, config):
    mask = (df['Pipeline'] == modele) & (df['Config'] == config)
    subset = df[mask]
    return {
        'F1_moyen': subset['F1_score'].mean() * 100,
        'Precision': subset['Precision'].mean() * 100,
        'Rappel': subset['Recall'].mean() * 100,
        'Nb_classes': len(subset)
    }

# Récupérer les statistiques pour chaque modèle
modeles = {
    'XGBoost (Base)': get_stats(df, 'XGBoost', 'Baseline'),
    'XGBoost + LDA (Base)': get_stats(df, 'XGBoost + LDA', 'Baseline'),
    'XGBoost + PCA (Base)': get_stats(df, 'XGBoost + PCA', 'Baseline'),
    'XGBoost (Optimisé)': {
        'F1_moyen': -opt_espece['best_score'] * 100 if 'best_score' in opt_espece else 'N/A',
        'Nb_classes': len(opt_espece.get('classes', []))
    },
    'XGBoost + LDA (Optimisé)': {
        'F1_moyen': 'À calculer',
        'Nb_classes': 'À vérifier'
    },
    'XGBoost + PCA (Optimisé)': {
        'F1_moyen': -opt_maladie['best_score'] * 100 if 'best_score' in opt_maladie else 'N/A',
        'Nb_classes': len(opt_maladie.get('classes', []))
    }
}

# Afficher le tableau récapitulatif
print("\nRÉCAPITULATIF DES 6 MODÈLES")
print("="*60)
print(f"{'Modèle':<25} | {'F1 moyen':<10} | {'Précision':<10} | {'Rappel':<10} | Classes")
print("-"*65)

for modele, stats in modeles.items():
    f1 = f"{stats['F1_moyen']:.2f}%" if isinstance(stats['F1_moyen'], float) else stats['F1_moyen']
    prec = f"{stats.get('Precision', 'N/A'):.2f}%" if 'Precision' in stats else 'N/A'
    rappel = f"{stats.get('Rappel', 'N/A'):.2f}%" if 'Rappel' in stats else 'N/A'
    classes = stats.get('Nb_classes', 'N/A')
    
    print(f"{modele:<25} | {f1:<10} | {prec:<10} | {rappel:<10} | {classes}")

print("\nLégende :")
print("- Base : Modèle avec paramètres par défaut")
print("- Optimisé : Après recherche d'hyperparamètres")
