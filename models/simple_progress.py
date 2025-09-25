import sys
import time

def simple_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Affiche une barre de progression simple
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print()

# Exemple d'utilisation
if __name__ == "__main__":
    items = list(range(100))
    total = len(items)
    
    print("Démonstration de la barre de progression :")
    for i, item in enumerate(items):
        # Faire quelque chose ici
        time.sleep(0.05)  # Simulation d'un traitement
        
        # Mettre à jour la barre de progression
        simple_progress_bar(i + 1, total, 
                          prefix='Progression:', 
                          suffix=f'Terminé ({i+1}/{total})', 
                          length=40)
