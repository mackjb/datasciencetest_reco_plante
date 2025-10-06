import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import os

def load_model(model_path, num_classes):
    """Charge le modèle entraîné"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def show_predictions(model, test_loader, class_names, num_images=5):
    """Affiche des exemples de prédictions"""
    plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 1:  # Prendre seulement un batch
                break
                
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for j in range(min(num_images, len(images))):
                plt.subplot(1, num_images, j+1)
                plt.xticks([])
                plt.yticks([])
                
                # Dénormaliser l'image
                img = images[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                plt.imshow(img)
                pred_label = class_names[preds[j]]
                true_label = class_names[labels[j]]
                color = 'green' if preds[j] == labels[j] else 'red'
                plt.xlabel(f'Prédit: {pred_label[:15]}...\nRéel: {true_label[:15]}...', 
                          color=color)
    plt.tight_layout()
    plt.show()

def evaluate(model, test_loader):
    """Évalue le modèle sur l'ensemble de test"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Précision sur l\'ensemble de test: {accuracy:.2f}%')
    return accuracy

def main():
    # Configuration
    DATA_DIR = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"
    MODEL_PATH = "resnet_plant_disease.pth"
    BATCH_SIZE = 32
    IMAGE_SIZE = 128
    
    # Transformation (doit correspondre à l'entraînement)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Charger les données
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    
    # Prendre un petit sous-ensemble pour l'évaluation
    indices = torch.randperm(len(full_dataset))[:500]  # 500 images pour l'évaluation
    eval_dataset = Subset(full_dataset, indices)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Charger le modèle
    num_classes = len(full_dataset.classes)
    model = load_model(MODEL_PATH, num_classes)
    
    # Afficher les métriques
    print("=== Évaluation du modèle ===")
    accuracy = evaluate(model, eval_loader)
    print("\n=== Exemples de prédictions ===")
    print("(Vert: correct, Rouge: incorrect)")
    show_predictions(model, eval_loader, full_dataset.classes)
    
    # Afficher la distribution des classes
    class_counts = {}
    for _, label in eval_dataset:
        class_name = full_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n=== Distribution des classes dans l'évaluation ===")
    for i, (class_name, count) in enumerate(class_counts.items()):
        print(f"{i+1}. {class_name}: {count} exemples")

if __name__ == "__main__":
    main()
