import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Données des modèles
data = [
    {'Modèle': 'XGBoost (Base)', 'Type': 'Base', 'F1': 81.07, 'Précision': 80.11, 'Rappel': 82.32, 'Classes': 21},
    {'Modèle': 'XGBoost + LDA (Base)', 'Type': 'Base', 'F1': 81.61, 'Précision': 79.93, 'Rappel': 83.71, 'Classes': 21},
    {'Modèle': 'XGBoost + PCA (Base)', 'Type': 'Base', 'F1': 86.67, 'Précision': 85.98, 'Rappel': 87.51, 'Classes': 21},
    {'Modèle': 'XGBoost (Optimisé)', 'Type': 'Optimisé', 'F1': 48.87, 'Précision': None, 'Rappel': None, 'Classes': 14},
    {'Modèle': 'XGBoost + LDA (Optimisé)', 'Type': 'Optimisé', 'F1': None, 'Précision': None, 'Rappel': None, 'Classes': None},
    {'Modèle': 'XGBoost + PCA (Optimisé)', 'Type': 'Optimisé', 'F1': 73.59, 'Précision': None, 'Rappel': None, 'Classes': 21}
]

df = pd.DataFrame(data)

# Créer le tableau de bord
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "table"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "pie"}]],
    subplot_titles=("Tableau des Performances", 
                   "Comparaison des F1",
                   "Précision vs Rappel",
                   "Répartition des Modèles")
)

# 1. Tableau des performances
trace1 = go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Modèle, df.Type, df.F1, df.Précision, df.Rappel, df.Classes],
               fill_color='lavender',
               align='left'))

# 2. Barres des scores F1
trace2 = go.Bar(
    x=df['Modèle'],
    y=df['F1'],
    name='Score F1',
    marker_color='indianred'
)

# 3. Nuage de points Précision vs Rappel
trace3 = go.Scatter(
    x=df['Précision'],
    y=df['Rappel'],
    mode='markers+text',
    text=df['Modèle'],
    marker=dict(size=12, color=df['F1'], colorscale='Viridis', showscale=True)
)

# 4. Camembert des types de modèles
trace4 = go.Pie(
    labels=['Base', 'Optimisé'],
    values=df.groupby('Type')['Modèle'].count(),
    name="Types de Modèles"
)

# Ajout des traces aux subplots
fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)
fig.add_trace(trace3, row=2, col=1)
fig.add_trace(trace4, row=2, col=2)

# Mise en page
fig.update_layout(
    height=1000,
    width=1200,
    title_text="Tableau de Bord des Performances des Modèles",
    showlegend=False
)

# Afficher le tableau de bord
fig.show()

# Sauvegarder le tableau de bord
fig.write_html("../results/dashboard_modeles.html")
print("Tableau de bord généré : results/dashboard_modeles.html")
