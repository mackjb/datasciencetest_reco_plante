import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# --- Configuration ---
DATA_DIR = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"
BATCH_SIZE = 32  # Augmenté pour accélérer
LEARNING_RATE = 0.001
NUM_WORKERS = 2
SUBSET_RATIO = 0.1  # 10% des données

# --- Transformation des images ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# --- Chargement du dataset ---
print("Chargement des données...")
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Création d'un sous-ensemble
indices = np.random.permutation(len(full_dataset))[:int(len(full_dataset) * SUBSET_RATIO)]
subset = Subset(full_dataset, indices)

# Séparation entraînement/validation
train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(subset, [train_size, val_size])

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                        shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                       shuffle=False, num_workers=NUM_WORKERS)

# --- Création du modèle ---
print("Initialisation du modèle...")
model = models.resnet50(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))

# --- Configuration de l'entraînement ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Boucle d'entraînement ---
print("\nDébut de l'entraînement (1 époque)...")
model.train()
running_loss = 0.0
correct = 0
total = 0

# Barre de progression pour l'entraînement
train_loop = tqdm(train_loader, desc="Entraînement")
for inputs, labels in train_loop:
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Réinitialisation des gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward pass et optimisation
    loss.backward()
    optimizer.step()
    
    # Statistiques
    running_loss += loss.item() * inputs.size(0)
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
    
    # Mise à jour de la barre de progression
    train_loop.set_postfix(loss=running_loss/total, accuracy=100.*correct/total)

# Calcul final
train_loss = running_loss / len(train_loader.dataset)
train_acc = 100. * correct / total

print(f'\nEntraînement terminé - Perte: {train_loss:.4f}, Précision: {train_acc:.2f}%')

# --- Validation ---
model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    val_loop = tqdm(val_loader, desc="Validation")
    for inputs, labels in val_loop:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        val_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        val_loop.set_postfix(loss=val_loss/total, accuracy=100.*correct/total)

val_loss = val_loss / len(val_loader.dataset)
val_acc = 100. * correct / total

print(f'\nValidation - Perte: {val_loss:.4f}, Précision: {val_acc:.2f}%')

# Sauvegarde du modèle
save_path = 'resnet_quick_test.pth'
torch.save(model.state_dict(), save_path)
print(f"\nModèle sauvegardé dans {save_path}")
