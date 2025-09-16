import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Données des modèles
data = {
    'Tâche': ['Espèces']*5 + ['Maladies']*5,
    'Modèle': [
        'LightGBM', 'Random Forest', 'AdaBoost', 'KNN', 'LDA',
        'LightGBM', 'Random Forest', 'AdaBoost', 'KNN', 'LDA'
    ],
    'F1-Score': [
        0.8997, 0.8718, 0.8420, 0.7680, 0.7421,  # Espèces
        0.8148, 0.7773, 0.7508, 0.6489, 0.6252   # Maladies
    ],
    'AUC': [
        0.9929, 0.9888, 0.9756, 0.9341, 0.0000,  # Espèces
        0.9867, 0.9796, 0.9632, 0.8952, 0.0000   # Maladies
    ]
}

df = pd.DataFrame(data)

# Configuration du style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

# Palette de couleurs
palette = sns.color_palette("husl", 5)

# Graphique F1-Score
plt.subplot(1, 2, 1)
sns.barplot(x='Tâche', y='F1-Score', hue='Modèle', data=df, palette=palette)
plt.title('Comparaison des Scores F1 par Modèle', pad=15, fontsize=14)
plt.ylim(0, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Graphique AUC
plt.subplot(1, 2, 2)
sns.barplot(x='Tâche', y='AUC', hue='Modèle', data=df, palette=palette)
plt.title('Comparaison des AUC par Modèle', pad=15, fontsize=14)
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustement et enregistrement
plt.tight_layout()
output_path = 'figures/comparaison_modeles_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graphique de comparaison enregistré sous : {output_path}")

# Création d'un graphique séparé pour les meilleurs modèles
plt.figure(figsize=(10, 6))
best_models = df[df['Modèle'].isin(['LightGBM', 'Random Forest', 'AdaBoost'])]
sns.barplot(x='Tâche', y='F1-Score', hue='Modèle', data=best_models, palette=palette[:3])
plt.title('Comparaison des 3 Meilleurs Modèles (F1-Score)', pad=15, fontsize=14)
plt.ylim(0.7, 0.95)
plt.tight_layout()
output_path_best = 'figures/meilleurs_modeles_comparison.png'
plt.savefig(output_path_best, dpi=300, bbox_inches='tight')
print(f"Graphique des meilleurs modèles enregistré sous : {output_path_best}")
