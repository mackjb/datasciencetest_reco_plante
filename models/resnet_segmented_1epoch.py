import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Configuration
DATA_DIR = "/workspaces/datasciencetest_reco_plante/dataset/plantvillage/data/plantvillage_5images/segmented"
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de {device} pour l'entraînement")

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Charger le dataset
try:
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    print(f"Nombre total d'images: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    
    # Diviser en ensembles d'entraînement et de validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Créer les DataLoaders avec moins de workers et une file d'attente plus petite
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=False)
    
    # Afficher des informations sur les données
    print(f"Images d'entraînement: {len(train_dataset)}")
    print(f"Images de validation: {len(val_dataset)}")
    print(f"Nombre de classes: {len(full_dataset.classes)}")

except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    print("Vérifiez le chemin du dataset et la structure des dossiers.")
    exit(1)

# Créer le modèle
def create_model(num_classes):
    model = models.resnet50(pretrained=True)
    
    # Geler les poids du modèle de base
    for param in model.parameters():
        param.requires_grad = False
    
    # Remplacer la dernière couche
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Initialiser le modèle
model = create_model(len(full_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Boucle d'entraînement
def train_model():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Entraînement"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Remise à zéro des gradients
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
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Validation
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistiques
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

# Entraînement pour une seule époque
print("\nDébut de l'entraînement...")
train_loss, train_acc = train_model()
val_loss, val_acc = validate()

# Afficher les résultats
print(f"\nRésultats après 1 époque:")
print(f"Entraînement - Perte: {train_loss:.4f}, Précision: {train_acc:.2f}%")
print(f"Validation - Perte: {val_loss:.4f}, Précision: {val_acc:.2f}%")

print("\nEntraînement terminé !")
