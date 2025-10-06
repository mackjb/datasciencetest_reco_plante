#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Affiche les résultats de l'optimisation XGBoost
"""

import joblib
from pathlib import Path
import pandas as pd

# Chemin des résultats
base_dir = Path("results/fast_optimization")

def load_and_display_results(target_type):
    """Charge et affiche les résultats pour un type de cible."""
    print(f"\n{'='*50}")
    print(f"RÉSULTATS POUR: {target_type.upper()}")
    print(f"{'='*50}")
    
    # Charger les résultats
    results_path = base_dir / f'results_{target_type}.pkl'
    results = joblib.load(results_path)
    
    # Afficher les métriques
    print("\nMÉTRIQUES:")
    print(f"- Meilleur score (log loss): {results['best_score']:.4f}")
    print(f"- Perte sur le test: {results['test_loss']:.4f}")
    
    # Afficher les meilleurs paramètres
    print("\nMEILLEURS PARAMÈTRES:")
    for param, value in results['best_params'].items():
        print(f"- {param}: {value:.4f}" if isinstance(value, float) else f"- {param}: {value}")
    
    # Afficher les caractéristiques les plus importantes
    print("\nTOP 10 CARACTÉRISTIQUES IMPORTANTES:")
    features = results['features']
    importances = results['feature_importance']
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    print(feature_importance.to_string(index=False))
    
    return results

if __name__ == "__main__":
    # Afficher les résultats pour les espèces et les maladies
    espece_results = load_and_display_results('espece')
    maladie_results = load_and_display_results('maladie')
