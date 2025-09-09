import pandas as pd
import plotly.express as px
from pathlib import Path

# Chemins des fichiers
results_dir = Path('/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_plante')
pred_file = results_dir / 'predicted_distribution_XGBoost_+_PCA_(Baseline).csv'
output_file = results_dir / 'rapport_prediction_especes.html'

# Lire les données
df = pd.read_csv(pred_file)
df = df.sort_values('Count', ascending=False)

# Créer le graphique
fig = px.bar(
    df, 
    x='Classe', 
    y='Count',
    title='Distribution des prédictions par espèce',
    labels={'Classe': 'Espèce', 'Count': 'Nombre de prédictions'},
    color='Count',
    color_continuous_scale='Blues'
)

# Mise en page
fig.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='',
    yaxis_title='Nombre de prédictions',
    coloraxis_showscale=False,
    height=600,
    margin=dict(b=150)
)

# Ajouter les valeurs sur les barres
fig.update_traces(
    texttemplate='%{y}',
    textposition='outside'
)

# Générer le contenu HTML
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Rapport des Prédictions par Espèce</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ 
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .table-responsive {{
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rapport des Prédictions par Espèce</h1>
            <p class="lead">Analyse des prédictions du modèle XGBoost + PCA (Baseline)</p>
        </div>
        
        <div class="summary-card">
            <h4>Résumé</h4>
            <p>Nombre total d'échantillons : <strong>{df['Count'].sum():,}</strong></p>
            <p>Nombre d'espèces uniques : <strong>{len(df)}</strong></p>
            <p>Espèce la plus fréquente : <strong>{df.iloc[0]['Classe']}</strong> ({df.iloc[0]['Count']} échantillons)</p>
        </div>
        
        <div id="chart"></div>
        
        <h3 style="margin-top: 40px;">Détail des prédictions</h3>
        <div class="table-responsive">
            {df.to_html(classes='table table-striped table-hover', index=False, float_format='{:.0f}'.format)}
        </div>
    </div>
    
    <script>
        var plotData = {fig.to_json()}
        Plotly.newPlot('chart', plotData.data, plotData.layout, {{responsive: true}});
        
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize('chart');
        }});
    </script>
</body>
</html>
'''

# Écrire le fichier HTML
with open(output_file, 'w') as f:
    f.write(html_content)

print(f'Rapport généré : {output_file}')
print(f'Accès direct : http://localhost:8000/nom_plante/rapport_prediction_especes.html')
