import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Charger les données
df = pd.read_csv('resultats_detailles.csv')

# Créer un rapport HTML
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Rapport d'Analyse du Modèle</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f1f1f1; padding: 20px; text-align: center; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric-box { 
            background-color: #f9f9f9; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 15px; 
            text-align: center;
            width: 30%;
        }
        .correct { color: green; }
        .incorrect { color: red; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Rapport d'Analyse du Modèle de Détection de Maladies des Plantes</h1>
        <p>Analyse des performances du modèle ResNet18</p>
    </div>
"""

# Calcul des métriques globales
total = len(df)
correct = df['Correct'].sum()
accuracy = (correct / total) * 100

# Ajout des métriques globales
html_content += f"""
<div class="section">
    <h2>Métriques Globales</h2>
    <div class="metrics">
        <div class="metric-box">
            <h3>Précision Globale</h3>
            <p style="font-size: 24px; font-weight: bold;">{accuracy:.2f}%</p>
        </div>
        <div class="metric-box">
            <h3>Prédictions Correctes</h3>
            <p style="font-size: 24px; font-weight: bold;">{correct}/{total}</p>
        </div>
        <div class="metric-box">
            <h3>Taux d'Erreur</h3>
            <p style="font-size: 24px; font-weight: bold;">{100-accuracy:.2f}%</p>
        </div>
    </div>
</div>
"""

# Analyse par classe
class_stats = df.groupby('True_Label').agg({
    'Correct': ['count', 'sum', lambda x: (x.sum()/x.count())*100]
}).reset_index()
class_stats.columns = ['Classe', 'Total', 'Correctes', 'Précision']
class_stats = class_stats.sort_values('Précision', ascending=True)

# Création du graphique des performances par classe
fig = px.bar(
    class_stats.tail(10),  # Top 10 meilleures classes
    x='Précision',
    y='Classe',
    orientation='h',
    title='Top 10 des Meilleures Classes (par précision)'
)
fig.write_html("top_classes.html")

# Graphique des pires classes
fig2 = px.bar(
    class_stats.head(10),  # Top 10 pires classes
    x='Précision',
    y='Classe',
    orientation='h',
    title='Top 10 des Pires Classes (par précision)'
)
fig2.write_html("worst_classes.html")

# Ajout des graphiques au rapport
html_content += """
<div class="section">
    <h2>Performances par Classe</h2>
    <iframe src="top_classes.html" width="100%" height="500px" style="border:none;"></iframe>
    <iframe src="worst_classes.html" width="100%" height="500px" style="border:none;"></iframe>
</div>
"""

# Exemples de prédictions
html_content += """
<div class="section">
    <h2>Exemples de Prédictions</h2>
    <table>
        <tr>
            <th>Vraie Classe</th>
            <th>Prédiction</th>
            <th>Confiance</th>
            <th>Résultat</th>
        </tr>
"""

# Ajout de 10 exemples de prédictions
for _, row in df.sample(10).iterrows():
    result_class = "correct" if row['Correct'] else "incorrect"
    result_text = "✅ Correct" if row['Correct'] else "❌ Incorrect"
    
    html_content += f"""
    <tr class="{result_class}">
        <td>{row['True_Label']}</td>
        <td>{row['Predicted_Label']}</td>
        <td>{row['Confidence']:.2f}</td>
        <td>{result_text}</td>
    </tr>
    """

html_content += """
    </table>
</div>
"""

# Recommandations
html_content += """
<div class="section">
    <h2>Recommandations pour l'Amélioration</h2>
    <h3>1. Augmenter les Données d'Entraînement</h3>
    <p>Pour les classes sous-performantes, collecter plus d'images d'entraînement.</p>
    
    <h3>2. Équilibrer le Jeu de Données</h3>
    <p>Certaines classes sont sous-représentées. Utiliser des techniques d'augmentation de données ou de rééquilibrage.</p>
    
    <h3>3. Affiner le Modèle</h3>
    <p>Entraîner le modèle plus longtemps ou ajuster le taux d'apprentissage.</p>
    
    <h3>4. Améliorer le Prétraitement</h3>
    <p>Ajouter des techniques d'augmentation de données plus avancées.</p>
</div>
"""

# Fin du document HTML
html_content += """
</body>
</html>
"""

# Sauvegarder le rapport
with open('rapport_analyse.html', 'w') as f:
    f.write(html_content)

print("Rapport généré avec succès : rapport_analyse.html")
