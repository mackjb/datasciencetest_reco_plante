import os
import sys
from html2image import Html2Image

def convert_html_to_png(html_path, output_dir):
    """Convertit un fichier HTML en image PNG"""
    try:
        # Initialiser Html2Image
        hti = Html2Image()
        
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemin de sortie pour l'image
        output_path = os.path.join(output_dir, f"lime_{os.path.basename(html_path).replace('.html', '.png')}")
        
        # Convertir le HTML en PNG
        hti.screenshot(
            html_file=html_path,
            save_as=os.path.basename(output_path),
            size=(1200, 800)  # Taille de la fenêtre
        )
        
        # Déplacer le fichier généré au bon endroit
        generated_file = os.path.join(os.getcwd(), os.path.basename(output_path))
        if os.path.exists(generated_file):
            os.rename(generated_file, output_path)
            print(f"Image sauvegardée : {output_path}")
            return True
        return False
        
    except Exception as e:
        print(f"Erreur lors de la conversion de {html_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_html_to_png.py <dossier_html> <dossier_sortie>")
        return
    
    html_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Trouver tous les fichiers HTML LIME
    html_files = [f for f in os.listdir(html_dir) if f.startswith('lime_') and f.endswith('.html')]
    
    if not html_files:
        print(f"Aucun fichier LIME trouvé dans {html_dir}")
        return
    
    # Convertir chaque fichier
    for html_file in html_files:
        html_path = os.path.join(html_dir, html_file)
        print(f"Traitement de {html_path}...")
        convert_html_to_png(html_path, output_dir)

if __name__ == "__main__":
    main()
