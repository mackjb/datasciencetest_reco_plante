#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualisation des résultats d'optimisation et des courbes d'apprentissage
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# Configuration du style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Chemins des résultats
base_dir = Path("results/fast_optimization")
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

def load_results():
    """Charge les résultats des optimisations."""
    results = {}
    for target in ['espece', 'maladie']:
        results[target] = joblib.load(base_dir / f'results_{target}.pkl')
    return results

def plot_feature_importance(results, target_type):
    """Affiche l'importance des caractéristiques."""
    data = results[target_type]
    df = pd.DataFrame({
        'Feature': data['features'],
        'Importance': data['feature_importance']
    }).sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title(f'Top 15 des caractéristiques importantes - {target_type.capitalize()}')
    plt.tight_layout()
    
    # Sauvegarde
    filename = output_dir / f'feature_importance_{target_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique d'importance des caractéristiques sauvegardé: {filename}")

def plot_learning_curves():
    """Affiche les courbes d'apprentissage."""
    # Note: À implémenter si les données d'entraînement sont disponibles
    pass

def create_results_table(results):
    """Crée un tableau des résultats."""
    data = []
    for target, res in results.items():
        data.append({
            'Cible': target.capitalize(),
            'Log Loss (test)': f"{res['test_loss']:.4f}",
            'Learning Rate': f"{res['best_params']['learning_rate']:.4f}",
            'Max Depth': res['best_params']['max_depth'],
            'Min Child Weight': res['best_params']['min_child_weight']
        })
    
    df = pd.DataFrame(data)
    
    # Affichage stylisé
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Création du tableau
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]  # Ajustement de la taille
    )
    
    # Mise en forme
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Sauvegarde
    filename = output_dir / 'results_summary.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tableau des résultats sauvegardé: {filename}")
    
    # Affichage dans la console
    print("\nRésumé des performances:")
    print(df.to_string(index=False))

def main():
    # Chargement des résultats
    results = load_results()
    
    # Création des visualisations
    for target in results.keys():
        plot_feature_importance(results, target)
    
    # Tableau des résultats
    create_results_table(results)
    
    print("\nVisualisations sauvegardées dans le dossier 'figures/'")

if __name__ == "__main__":
    main()
