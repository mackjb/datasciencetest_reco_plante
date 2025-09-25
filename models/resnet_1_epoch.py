import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from PIL import Image

# --- Configuration ---
DATA_DIR = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage dataset/segmented"  # Chemin vers le dataset complet
BATCH_SIZE = 16  # Réduit pour économiser la mémoire
NUM_EPOCHS = 1    # Une seule époque comme demandé
LEARNING_RATE = 0.001
NUM_WORKERS = 2  # Réduit le nombre de workers

# --- Transformation des images ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# --- Chargement du dataset ---
# Vérification du dossier de données
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Le dossier de données {DATA_DIR} n'existe pas.")

# Création du dataset
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

# --- Création du modèle ---
model = models.resnet50(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))  # Nombre de classes du dataset

# --- Configuration de l'entraînement ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Boucle d'entraînement ---
print("Début de l'entraînement...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Réinitialisation des gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    # Calcul de la perte moyenne pour l'époque
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Époque {epoch+1}/{NUM_EPOCHS}, Perte: {epoch_loss:.4f}')

print("Entraînement terminé !")
