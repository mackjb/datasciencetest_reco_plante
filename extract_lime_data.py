import os
import json
import base64
import re
from bs4 import BeautifulSoup

def extract_lime_data(html_path):
    """Extrait les données LIME du fichier HTML"""
    try:
        # Lire le contenu du fichier
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Afficher les premières et dernières lignes pour le débogage
        print("\n=== Début du fichier HTML ===")
        print(content[:500])
        print("\n=== Fin du fichier HTML ===")
        print(content[-500:])
        
        # Essayer d'extraire les données JSON
        soup = BeautifulSoup(content, 'html.parser')
        
        # Rechercher tous les scripts
        scripts = soup.find_all('script')
        print(f"\nNombre de scripts trouvés: {len(scripts)}")
        
        # Afficher les 5 premiers scripts pour inspection
        for i, script in enumerate(scripts[:5]):
            script_content = script.string or ''
            print(f"\n--- Script {i+1} (longueur: {len(script_content)}) ---")
            print(script_content[:200] + '...' if len(script_content) > 200 else script_content)
        
        # Essayer de trouver des données JSON dans le contenu
        json_matches = re.findall(r'\{.*\}', content, re.DOTALL)
        print(f"\nNombre de correspondances JSON trouvées: {len(json_matches)}")
        
        # Afficher les premières correspondances
        for i, match in enumerate(json_matches[:3]):
            print(f"\n--- Correspondance JSON {i+1} ---")
            print(match[:200] + '...' if len(match) > 200 else match)
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'extraction des données: {str(e)}")
        return False

def main():
    input_html = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_plante/interpretability/lime_example_1.html'
    
    print(f"Analyse du fichier: {input_html}")
    print(f"Taille du fichier: {os.path.getsize(input_html) / 1024:.1f} KB")
    
    if not os.path.exists(input_html):
        print("Erreur: Le fichier n'existe pas")
        return
    
    extract_lime_data(input_html)

if __name__ == "__main__":
    main()
