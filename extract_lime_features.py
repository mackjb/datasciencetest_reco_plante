import os
import re

def extract_lime_features(html_path):
    """Extrait les caractéristiques LIME du fichier HTML"""
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Rechercher les lignes contenant les caractéristiques
        pattern = r'\["([^"]+)",\s*"([^"]+)",\s*([-\d.]+)\]'
        matches = re.findall(pattern, content)
        
        if not matches:
            print("Aucune caractéristique trouvée dans le fichier")
            return []
        
        # Trier par importance (valeur absolue)
        features = []
        for match in matches:
            name = match[0]
            value = float(match[1])
            importance = float(match[2])
            features.append({
                'name': name,
                'value': value,
                'importance': importance
            })
        
        # Trier par importance absolue décroissante
        features.sort(key=lambda x: abs(x['importance']), reverse=True)
        return features
        
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques: {str(e)}")
        return []

def plot_features(features, output_path, top_n=10):
    """Crée un graphique des caractéristiques les plus importantes"""
    import matplotlib.pyplot as plt
    
    if not features:
        print("Aucune donnée à afficher")
        return False
    
    try:
        # Prendre les top_n caractéristiques
        top_features = features[:top_n]
        names = [f['name'] for f in top_features]
        importances = [f['importance'] for f in top_features]
        colors = ['green' if x > 0 else 'red' for x in importances]
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        plt.barh(names, importances, color=colors)
        plt.title('Top 10 des caractéristiques les plus importantes')
        plt.xlabel('Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Sauvegarder le graphique
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique sauvegardé: {output_path}")
        return True
        
    except Exception as e:
        print(f"Erreur lors de la création du graphique: {str(e)}")
        return False

def main():
    input_html = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/nom_plante/interpretability/lime_example_1.html'
    output_dir = '/tmp/lime_analysis'
    output_plot = os.path.join(output_dir, 'top_features.png')
    
    print(f"Analyse du fichier: {input_html}")
    
    # Extraire les caractéristiques
    features = extract_lime_features(input_html)
    
    if not features:
        print("Aucune caractéristique n'a pu être extraite")
        return
    
    # Afficher les 10 premières caractéristiques
    print("\nTop 10 des caractéristiques les plus importantes:")
    print("-" * 80)
    print(f"{'Caractéristique':<30} {'Valeur':<15} {'Importance':<15}")
    print("-" * 80)
    for feat in features[:10]:
        print(f"{feat['name']:<30} {feat['value']:<15.4f} {feat['importance']:<15.4f}")
    
    # Créer et sauvegarder le graphique
    if plot_features(features, output_plot):
        print(f"\nVisualisation sauvegardée: {output_plot}")
    else:
        print("\nÉchec de la création de la visualisation")

if __name__ == "__main__":
    main()
