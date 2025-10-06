import pandas as pd

# Charger les données
df = pd.read_csv('../results/models/xgboost/class_results.csv')

# Définir les configurations de base
configs = {
    'XGBoost': 'Baseline',
    'XGBoost + LDA': 'Baseline',
    'XGBoost + PCA': 'Baseline'
}

# Afficher les résultats pour chaque modèle
for modele, config in configs.items():
    print(f"\n{'='*80}")
    print(f"MODÈLE: {modele} ({config})")
    print("="*80)
    
    # Filtrer les données
    mask = (df['Pipeline'] == modele) & (df['Config'] == config)
    resultats = df[mask].sort_values('F1_score', ascending=False)
    
    # Afficher le tableau formaté
    print("| {:<40} | {:<10} | {:<10} | {:<10} | {:<8} |".format(
        "Classe", "Précision", "Rappel", "F1-score", "Support"))
    print("|{0:42}|{0:12}|{0:12}|{0:12}|{0:10}|".format("-"*10))
    
    for _, row in resultats.iterrows():
        print("| {:<40} | {:>9.2f}% | {:>8.2f}% | {:>8.2f}% | {:>7} |".format(
            row['Classe'], 
            row['Precision']*100, 
            row['Recall']*100,
            row['F1_score']*100,
            int(row['Support'])
        ))
    
    # Afficher les moyennes
    print("\nMoyennes :")
    print(f"- Précision : {resultats['Precision'].mean()*100:.2f}%")
    print(f"- Rappel : {resultats['Recall'].mean()*100:.2f}%")
    print(f"- F1-score : {resultats['F1_score'].mean()*100:.2f}%")
    print("\n\n")
