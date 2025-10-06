import os
import sys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def convert_lime_to_png(html_path, output_dir):
    """Convertit un fichier HTML LIME en image PNG"""
    try:
        # Configuration de Selenium
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Initialiser le navigateur
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # Charger la page HTML
        driver.get(f"file://{os.path.abspath(html_path)}")
        
        # Attendre que la page soit chargée
        driver.implicitly_wait(10)
        
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemin de sortie pour l'image
        output_path = os.path.join(output_dir, f"lime_{os.path.basename(html_path).replace('.html', '.png')}")
        
        # Prendre une capture d'écran
        driver.save_screenshot(output_path)
        print(f"Image sauvegardée : {output_path}")
        
        # Fermer le navigateur
        driver.quit()
        return True
        
    except Exception as e:
        print(f"Erreur lors de la conversion de {html_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_lime_to_png.py <dossier_html> <dossier_sortie>")
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
        convert_lime_to_png(html_path, output_dir)

if __name__ == "__main__":
    main()
