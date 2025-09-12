import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer le dossier pour les figures si nécessaire
os.makedirs('../figures', exist_ok=True)

# Données des performances
data = {
    'Modèle': ['XGBoost + LDA', 'XGBoost + PCA', 'XGBoost', 'XGBoost + PCA', 'XGBoost + LDA', 'XGBoost'],
    'Type': ['Espèces', 'Espèces', 'Espèces', 'Maladies', 'Maladies', 'Maladies'],
    'F1': [85.92, 85.54, 83.00, 79.81, 75.90, 74.59],
    'Précision': [84.65, 84.15, 81.51, 78.53, 74.00, 73.71],
    'Rappel': [87.49, 87.88, 85.39, 82.54, 79.76, 77.43]
}

df = pd.DataFrame(data)

# Style des graphiques
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# 1. Comparaison des modèles par type
plt.figure(figsize=(12, 6))
sns.barplot(x='Modèle', y='F1', hue='Type', data=df)
plt.title('Comparaison des modèles par score F1')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../figures/comparaison_modeles.png')

# 2. Top 5 des espèces et maladies
plt.figure(figsize=(14, 12))

# Données des espèces
especes_data = {
    'Classe': ['Maïs', 'Soja', 'Orange', 'Courge', 'Tomate'],
    'F1': [94.41, 91.60, 91.07, 89.19, 88.18],
    'Type': 'Espèce'
}

# Données des maladies
maladies_data = {
    'Classe': ['Sain', 'Citrus Greening', 'Feuilles Jaunes', 'Rouille', 'Oïdium'],
    'F1': [100.0, 90.63, 90.51, 89.05, 82.61],
    'Type': 'Maladie'
}

df_top = pd.concat([
    pd.DataFrame(especes_data),
    pd.DataFrame(maladies_data)
])

# Graphique des performances par classe
g = sns.catplot(x='F1', y='Classe', col='Type', data=df_top, kind='bar', 
                height=6, aspect=0.7, sharex=False)
g.set_titles("{col_name}")
g.fig.suptitle('Top 5 des performances par catégorie')
plt.tight_layout()
plt.savefig('../figures/top5_performances.png')

# 3. Heatmap de corrélation entre métriques
plt.figure(figsize=(8, 6))
corr = df[['F1', 'Précision', 'Rappel']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Corrélation entre les métriques')
plt.tight_layout()
plt.savefig('../figures/correlation_metriques.png')

print("Visualisations générées avec succès dans le dossier 'figures' !")
