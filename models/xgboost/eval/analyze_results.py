import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix

def load_model(model_path, num_classes):
    """Charge le modèle entraîné"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_results_table(model, test_loader, class_names):
    """Génère un tableau des résultats détaillés"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
    
    # Créer un DataFrame avec les résultats
    results = []
    for i, (pred, label, prob) in enumerate(zip(all_preds, all_labels, all_probs)):
        results.append({
            'Image_ID': i,
            'True_Label': class_names[label],
            'Predicted_Label': class_names[pred],
            'Confidence': prob[pred],
            'Correct': pred == label
        })
    
    df_results = pd.DataFrame(results)
    
    # Calculer les métriques globales
    accuracy = (df_results['Correct'].mean()) * 100
    
    # Générer le rapport de classification
    print("\n=== RAPPORT DE CLASSIFICATION ===")
    print(classification_report(
        df_results['True_Label'], 
        df_results['Predicted_Label'],
        target_names=class_names,
        zero_division=0
    ))
    
    # Afficher les statistiques globales
    print("\n=== STATISTIQUES GLOBALES ===")
    print(f"Précision globale: {accuracy:.2f}%")
    
    # Afficher les classes les plus difficiles
    errors = df_results[df_results['Correct'] == False]
    if not errors.empty:
        print("\n=== CLASSES LES PLUS DIFFICILES (plus d'erreurs) ===")
        error_counts = errors.groupby('True_Label').size().sort_values(ascending=False)
        print(error_counts.head(10).to_string())
    
    return df_results

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Affiche la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    DATA_DIR = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"
    MODEL_PATH = "resnet_plant_disease.pth"
    BATCH_SIZE = 32
    IMAGE_SIZE = 128
    
    # Transformation
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Charger les données
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    
    # Prendre un sous-ensemble pour l'évaluation
    indices = torch.randperm(len(full_dataset))[:1000]  # 1000 images pour l'évaluation
    eval_dataset = Subset(full_dataset, indices)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Charger le modèle
    num_classes = len(full_dataset.classes)
    model = load_model(MODEL_PATH, num_classes)
    
    # Générer et afficher les résultats
    df_results = generate_results_table(model, eval_loader, full_dataset.classes)
    
    # Afficher un échantillon des prédictions
    print("\n=== EXEMPLE DE PRÉDICTIONS ===")
    print(df_results[['True_Label', 'Predicted_Label', 'Confidence', 'Correct']].head(20).to_string())
    
    # Sauvegarder les résultats dans un fichier CSV
    df_results.to_csv('resultats_detailles.csv', index=False)
    print("\nRésultats détaillés sauvegardés dans 'resultats_detailles.csv'")
    
    # Afficher la matrice de confusion
    plot_confusion_matrix(df_results['True_Label'], 
                         df_results['Predicted_Label'],
                         full_dataset.classes)

if __name__ == "__main__":
    main()
