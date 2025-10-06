import os
import re
import json
import base64
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def parse_lime_html(html_path):
    """Parse le fichier HTML LIME et extrait les données pertinentes"""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Utiliser BeautifulSoup pour parser le HTML
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extraire les données JSON du script
    script_content = soup.find('script', string=re.compile('var data = '))
    if not script_content:
        print("Données LIME non trouvées dans le fichier HTML")
        return None
    
    # Extraire la chaîne JSON
    json_str = re.search(r'var data = (\[.*?\]);', script_content.string, re.DOTALL)
    if not json_str:
        print("Format de données inattendu dans le fichier HTML")
        return None
    
    try:
        data = json.loads(json_str.group(1))
        return data
    except json.JSONDecodeError as e:
        print(f"Erreur lors du décodage des données JSON: {e}")
        return None

def plot_lime_data(data, output_path):
    """Crée un graphique à partir des données LIME"""
    if not data or 'features' not in data:
        print("Données LIME incomplètes")
        return False
    
    try:
        features = data['features']
        
        # Extraire les noms des caractéristiques et leurs valeurs
        feat_names = [f['name'] for f in features]
        feat_values = [f['value'] for f in features]
        
        # Créer un graphique à barres
        plt.figure(figsize=(12, 6))
        colors = ['green' if v > 0 else 'red' for v in feat_values]
        plt.barh(feat_names, feat_values, color=colors)
        
        plt.title('Importance des caractéristiques LIME')
        plt.xlabel('Valeur')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Sauvegarder le graphique
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique sauvegardé: {output_path}")
        return True
        
    except Exception as e:
        print(f"Erreur lors de la création du graphique: {e}")
        return False

def main():
    # Chemins d'entrée et de sortie
    input_html = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_plante/interpretability/lime_example_1.html'
    output_dir = '/tmp/lime_plots'
    output_png = os.path.join(output_dir, 'lime_plot.png')
    
    # Parser le HTML
    lime_data = parse_lime_html(input_html)
    if not lime_data:
        print("Impossible d'extraire les données LIME")
        return
    
    # Créer le graphique
    if plot_lime_data(lime_data, output_png):
        print(f"\nVisualisation LIME créée avec succès!")
        print(f"Fichier: {output_png}")
    else:
        print("Échec de la création de la visualisation")

if __name__ == "__main__":
    main()
