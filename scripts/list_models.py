import pandas as pd
from pycaret.classification import *

# Créer une expérience factice
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'target': [0, 1, 0, 1, 0]
})

# Initialiser l'environnement PyCaret
exp = setup(data, target='target', verbose=False)

# Afficher les modèles disponibles
print("Colonnes disponibles dans le DataFrame des modèles:")
models_list = models()
print("\nColonnes:", models_list.columns.tolist())
print("\nModèles disponibles dans PyCaret:")
print(models_list[['Reference', 'Name']].to_string(index=False))
