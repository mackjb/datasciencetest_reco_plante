import pandas as pd
from pathlib import Path

def load_and_process_data(filepath, target_col):
    """Charge et traite les données de résultats."""
    df = pd.read_csv(filepath)
    df['Pipeline_Config'] = df['Pipeline'] + ' (' + df['Config'] + ')'
    return df.sort_values(['Pipeline_Config', 'F1_score'], ascending=[True, False])

def generate_report(df, target_name, output_path):
    """Génère un rapport HTML détaillé des scores."""
    # Formater les valeurs numériques
    df['Support'] = df['Support'].astype(int)
    for col in ['Precision', 'Recall', 'F1_score']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    # Générer le HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scores détaillés - {target_name}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .table {{ font-size: 14px; }}
            .table thead th {{ 
                background-color: #3498db; 
                color: white; 
                position: sticky;
                top: 0;
            }}
            .pipeline-header {{
                background-color: #f2f9ff;
                font-weight: bold;
            }}
            .table-container {{
                max-height: 600px;
                overflow-y: auto;
                margin-bottom: 30px;
            }}
            .metrics-box {{
                background: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Scores détaillés - {target_name}</h1>
            <p class="lead">Précision, Rappel et F1-score par classe et par configuration de pipeline</p>
            
            <div class="table-container">
                <table class="table table-striped table-hover">
                    <thead>
                        <tr>
                            <th>Pipeline (Config)</th>
                            <th>Classe</th>
                            <th>Précision</th>
                            <th>Rappel</th>
                            <th>F1-Score</th>
                            <th>Support</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Ajouter les lignes du tableau
    current_pipeline = None
    for _, row in df.iterrows():
        if row['Pipeline_Config'] != current_pipeline:
            current_pipeline = row['Pipeline_Config']
            html += f"""
                    <tr class="pipeline-header">
                        <td colspan="6">{current_pipeline}</td>
                    </tr>
            """
        
        html += f"""
                    <tr>
                        <td></td>
                        <td>{row['Classe']}</td>
                        <td>{row['Precision']}</td>
                        <td>{row['Recall']}</td>
                        <td><strong>{row['F1_score']}</strong></td>
                        <td>{row['Support']}</td>
                    </tr>
        """
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Sauvegarder le rapport
    with open(output_path, 'w') as f:
        f.write(html)

def main():
    base_dir = Path('/workspaces/datasciencetest_reco_plante/results/models/xgboost')
    
    # Traiter les données pour les maladies
    maladie_file = base_dir / 'nom_maladie/class_results.csv'
    if maladie_file.exists():
        df_maladie = load_and_process_data(maladie_file, 'Maladie')
        generate_report(df_maladie, 'Classification des Maladies', 
                       base_dir / 'nom_maladie/detailed_scores_maladies.html')
    
    # Traiter les données pour les espèces
    plante_file = base_dir / 'nom_plante/class_results.csv'
    if plante_file.exists():
        df_plante = load_and_process_data(plante_file, 'Espèce')
        generate_report(df_plante, 'Classification des Espèces',
                       base_dir / 'nom_plante/detailed_scores_especes.html')
    
    # Créer une page d'index
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapports des Scores Détailés</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 40px; background-color: #f8f9fa; }}
            .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .card-header {{ background-color: #3498db; color: white; }}
            .card-body {{ background-color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-5">Rapports des Scores Détailés</h1>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Classification des Maladies</h3>
                        </div>
                        <div class="card-body text-center">
                            <p>Scores détaillés pour chaque maladie et chaque configuration de pipeline</p>
                            <a href="nom_maladie/detailed_scores_maladies.html" class="btn btn-primary">Voir le rapport</a>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Classification des Espèces</h3>
                        </div>
                        <div class="card-body text-center">
                            <p>Scores détaillés pour chaque espèce et chaque configuration de pipeline</p>
                            <a href="nom_plante/detailed_scores_especes.html" class="btn btn-primary">Voir le rapport</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(base_dir / 'index_scores_detailles.html', 'w') as f:
        f.write(index_html)
    
    print("Rapports générés avec succès !")
    print(f"Page d'accueil : http://localhost:8000/index_scores_detailles.html")

if __name__ == "__main__":
    main()
