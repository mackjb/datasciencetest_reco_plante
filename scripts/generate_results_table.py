import pandas as pd
import os

def load_and_format(filepath, target_name):
    """Charge et formate les résultats d'une cible spécifique."""
    df = pd.read_csv(filepath)
    df = df.sort_values('Test_F1_weighted', ascending=False)
    df = df.head(3)  # Top 3 modèles
    df['Cible'] = target_name
    return df[['Cible', 'Pipeline', 'Config', 'Test_Accuracy', 'Test_F1_weighted']]

def generate_html(df, title):
    """Génère le HTML pour un tableau de résultats."""
    html = df.to_html(
        index=False,
        classes='table table-striped table-hover table-bordered',
        float_format=lambda x: f'{x:.4f}',
        border=0
    )
    return f'''
    <div class="container mb-5">
        <h2 class="mb-3">{title}</h2>
        <div class="table-responsive">
            {html}
        </div>
    </div>
    '''

def generate_single_report(df, title, output_path):
    """Génère un rapport HTML pour une seule cible."""
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{ 
                padding: 20px; 
                background-color: #f8f9fa; 
            }}
            h1, h2 {{ 
                color: #2c3e50; 
                margin-top: 20px; 
            }}
            .table {{ 
                font-size: 14px; 
                box-shadow: 0 0 10px rgba(0,0,0,0.05); 
            }}
            .table thead th {{ 
                background-color: #3498db; 
                color: white; 
                border: none;
            }}
            .table tbody tr:nth-child(even) {{ 
                background-color: #f2f9ff; 
            }}
            .table tbody tr:hover {{ 
                background-color: #e1f0fa; 
            }}
            .container {{ 
                background: white;
                padding: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 20px auto;
                max-width: 1000px;
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
            <h1 class="text-center mb-4">{title}</h1>
            {df.to_html(
                index=False,
                classes='table table-striped table-hover table-bordered',
                float_format=lambda x: f'{x:.4f}',
                border=0
            )}
            <div class="metrics-box">
                <h4>Métriques clés :</h4>
                <p><strong>Meilleur modèle :</strong> {df.iloc[0]['Pipeline']} ({df.iloc[0]['Config']})</p>
                <p><strong>Précision :</strong> {df.iloc[0]['Test_Accuracy']:.4f}</p>
                <p><strong>Score F1 pondéré :</strong> {df.iloc[0]['Test_F1_weighted']:.4f}</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Rapport généré : {output_path}')

def main():
    # Chemins des fichiers
    base_dir = '/workspaces/datasciencetest_reco_plante/results/models/xgboost'
    
    # Générer le rapport pour les maladies
    df_maladie = load_and_format(
        os.path.join(base_dir, 'nom_maladie/global_results.csv'),
        'Maladies'
    )
    generate_single_report(
        df_maladie,
        'Rapport des Performances - Classification des Maladies',
        os.path.join(base_dir, 'rapport_maladies.html')
    )
    
    # Générer le rapport pour les espèces
    df_plante = load_and_format(
        os.path.join(base_dir, 'nom_plante/global_results.csv'),
        'Espèces'
    )
    generate_single_report(
        df_plante,
        'Rapport des Performances - Classification des Espèces',
        os.path.join(base_dir, 'rapport_especes.html')
    )
    
    # Créer une page d'index
    with open(os.path.join(base_dir, 'index_rapports.html'), 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapports de Classification</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{ padding: 40px; background-color: #f8f9fa; }}
                .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .card-header {{ background-color: #3498db; color: white; }}
                .card-body {{ background-color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center mb-5">Rapports de Classification</h1>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h3>Classification des Maladies</h3>
                            </div>
                            <div class="card-body text-center">
                                <p>Analyse des performances pour la détection des maladies des plantes</p>
                                <a href="rapport_maladies.html" class="btn btn-primary">Voir le rapport</a>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h3>Classification des Espèces</h3>
                            </div>
                            <div class="card-body text-center">
                                <p>Analyse des performances pour l'identification des espèces de plantes</p>
                                <a href="rapport_especes.html" class="btn btn-primary">Voir le rapport</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        ''')
    
    print("\nAccédez aux rapports :")
    print(f"- Page d'accueil: http://localhost:8000/index_rapports.html")
    print(f"- Rapport Maladies: http://localhost:8000/rapport_maladies.html")
    print(f"- Rapport Espèces: http://localhost:8000/rapport_especes.html")

if __name__ == "__main__":
    main()
