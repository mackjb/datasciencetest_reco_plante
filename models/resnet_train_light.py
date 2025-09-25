import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# --- Configuration ---
DATA_DIR = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"
MODEL_SAVE_PATH = "resnet_plant_disease.pth"
BATCH_SIZE = 16
IMAGE_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 1
SUBSET_RATIO = 0.2  # 20% des données

# --- Transformation des images ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),  # Légère augmentation des données
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def main():
    print("=== Configuration ===")
    print(f"Taille de l'image: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Taille du lot: {BATCH_SIZE}")
    print(f"Taux d'apprentissage: {LEARNING_RATE}")
    print(f"Sous-ensemble des données: {SUBSET_RATIO*100}%")
    
    # --- Chargement des données ---
    print("\nChargement des données...")
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        
        # Création d'un sous-ensemble
        subset_size = int(len(full_dataset) * SUBSET_RATIO)
        indices = torch.randperm(len(full_dataset))[:subset_size]
        subset = torch.utils.data.Subset(full_dataset, indices)
        
        # Séparation entraînement/validation (80/20)
        train_size = int(0.8 * len(subset))
        val_size = len(subset) - train_size
        train_dataset, val_dataset = random_split(subset, [train_size, val_size])
        
        # Création des DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
        
        print(f"\nDonnées chargées avec succès:")
        print(f"- Total d'images: {len(full_dataset)}")
        print(f"- Sous-ensemble: {len(subset)} images")
        print(f"  - Entraînement: {len(train_dataset)}")
        print(f"  - Validation: {len(val_dataset)}")
        print(f"- Nombre de classes: {len(full_dataset.classes)}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        return
    
    # --- Initialisation du modèle ---
    print("\nInitialisation du modèle ResNet18...")
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Modèle chargé sur {device}")
    
    # --- Boucle d'entraînement ---
    print(f"\nDébut de l'entraînement pour {EPOCHS} époque(s)...")
    
    for epoch in range(EPOCHS):
        # Entraînement
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_loader, desc=f"Époque {epoch+1}/{EPOCHS} [Entraînement]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistiques
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Mise à jour de la barre de progression
            train_loop.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Époque {epoch+1}/{EPOCHS} [Validation]")
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_loop.set_postfix({
                    'val_loss': val_loss/val_total,
                    'val_acc': 100.*val_correct/val_total
                })
        
        # Affichage des résultats de l'époque
        print(f"\nRésultats de l'époque {epoch+1}:")
        print(f"Entraînement - Perte: {running_loss/len(train_dataset):.4f}, Précision: {100.*correct/len(train_dataset):.2f}%")
        print(f"Validation - Perte: {val_loss/len(val_dataset):.4f}, Précision: {100.*val_correct/len(val_dataset):.2f}%")
    
    # Sauvegarde du modèle
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModèle sauvegardé dans {MODEL_SAVE_PATH}")
    print("Entraînement terminé avec succès !")

if __name__ == "__main__":
    main()
