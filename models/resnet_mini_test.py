import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# --- Configuration minimale ---
DATA_DIR = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"
BATCH_SIZE = 4  # Très petit batch size
NUM_CLASSES = 38  # Nombre de classes dans PlantVillage

# Transformation minimale
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Très petite taille
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

try:
    # Chargement d'un petit sous-ensemble
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    
    # Prendre seulement 100 images pour le test
    indices = torch.randperm(len(dataset))[:100]
    mini_dataset = Subset(dataset, indices)
    
    # Création du DataLoader
    loader = DataLoader(mini_dataset, batch_size=BATCH_SIZE, 
                       shuffle=True, num_workers=0)  # Pas de workers
    
    # Modèle simplifié
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Configuration basique
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entraînement d'une seule boucle
    print("Test en cours...")
    model.train()
    for inputs, labels in tqdm(loader, desc="Test"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print("\nTest réussi ! Le modèle peut s'entraîner.")
    print("Vous pouvez maintenant essayer avec plus de données.")
    
except Exception as e:
    print(f"\nErreur: {str(e)}")
    print("\nConseils:")
    print("1. Vérifiez le chemin des données")
    print("2. Réduisez encore la taille du batch ou des images")
    print("3. Vérifiez les permissions des dossiers")
