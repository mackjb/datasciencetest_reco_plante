import time
import sys

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

print("=== Démonstration de la barre de progression ===\n")

# Simulation d'un traitement long
print("Simulation d'un traitement...")
total_items = 100
for i in range(total_items + 1):
    # Simulation d'un traitement
    time.sleep(0.05)
    
    # Mise à jour de la barre de progression
    simple_progress_bar(
        i, 
        total_items,
        prefix='Progression:',
        suffix=f'[{i}/{total_items}] - Traitement en cours...',
        length=40
    )

print("\nTraitement terminé avec succès !")
