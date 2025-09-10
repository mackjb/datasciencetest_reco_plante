import os
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import joblib
import json

def load_lime_data(interpretability_dir):
    """Charge les données nécessaires pour générer les explications LIME"""
    try:
        # Charger les données du fichier xai_summary.json
        with open(os.path.join(interpretability_dir, 'xai_summary.json'), 'r') as f:
            data = json.load(f)
        
        # Charger le modèle et les explications LIME
        model_path = os.path.join(interpretability_dir, 'lime_explainer.joblib')
        if not os.path.exists(model_path):
            print(f"Erreur: Fichier {model_path} non trouvé")
            return None
            
        explainer = joblib.load(model_path)
        
        # Charger un exemple d'image
        example_path = os.path.join(interpretability_dir, 'example_image.npy')
        if os.path.exists(example_path):
            example_image = np.load(example_path)
        else:
            print("Avertissement: Exemple d'image non trouvé, utilisation d'une image aléatoire")
            example_image = np.random.rand(224, 224, 3)
            
        return {
            'explainer': explainer,
            'example_image': example_image,
            'class_names': data.get('class_names', []),
            'target_class': data.get('target_class', 0)
        }
        
    except Exception as e:
        print(f"Erreur lors du chargement des données LIME: {str(e)}")
        return None

def generate_lime_visualization(data, output_path):
    """Génère une visualisation LIME à partir des données chargées"""
    try:
        explainer = data['explainer']
        image = data['example_image']
        
        # Générer l'explication
        explanation = explainer.explain_instance(
            image,
            classifier_fn=data.get('classifier_fn', None),
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )
        
        # Créer la visualisation
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        # Afficher l'image avec les explications
        plt.figure(figsize=(10, 8))
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f"Explication LIME pour la classe: {data['class_names'][data['target_class']]}")
        plt.axis('off')
        
        # Sauvegarder l'image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Visualisation LIME sauvegardée: {output_path}")
        return True
        
    except Exception as e:
        print(f"Erreur lors de la génération de la visualisation LIME: {str(e)}")
        return False

def main():
    # Dossiers d'entrée et de sortie
    base_dir = '/workspaces/datasciencetest_reco_plante/results/models/xgboost/'
    output_dir = '/tmp/lime_visualizations/'
    
    # Pour chaque dossier d'interprétabilité
    for model_type in ['nom_plante', 'nom_maladie']:
        interpretability_dir = os.path.join(base_dir, model_type, 'interpretability')
        
        if not os.path.exists(interpretability_dir):
            print(f"Dossier non trouvé: {interpretability_dir}")
            continue
            
        print(f"\nTraitement du dossier: {interpretability_dir}")
        
        # Charger les données LIME
        lime_data = load_lime_data(interpretability_dir)
        if not lime_data:
            print(f"Impossible de charger les données LIME pour {model_type}")
            continue
            
        # Générer la visualisation
        output_path = os.path.join(output_dir, f"lime_visualization_{model_type}.png")
        if generate_lime_visualization(lime_data, output_path):
            print(f"Visualisation générée avec succès: {output_path}")
        else:
            print(f"Échec de la génération de la visualisation pour {model_type}")

if __name__ == "__main__":
    main()
