# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Importa las clases del modelo y utilidades
from src.model import VisionTransformer
from src.utils import get_mnist_transform, IMAGE_SIZE # También puedes usar MNIST_MEAN, MNIST_STD

# --- Parámetros del modelo ---
PATCH_SIZE = 7
IN_CHANNELS = 1
EMBED_DIM = 128
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4
NUM_CLASSES = 10

# Configuración del dispositivo (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en {device}...")

# 1. Preparación de datos
train_transform = get_mnist_transform(is_train=True)

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Instanciar el modelo
model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, EMBED_DIM, 
                          NUM_HEADS, NUM_TRANSFORMER_LAYERS, NUM_CLASSES)
model.to(device)

# 3. Configuración del entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Bucle de entrenamiento
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    print(f'Epoch {epoch+1} completo. Pérdida promedio: {running_loss / len(train_loader):.4f}')

    # Evaluación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Precisión en el conjunto de prueba: {accuracy:.2f}%\n')

# 5. Guardar el modelo entrenado
model_save_path = 'models/vit_mnist_model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Crea la carpeta 'models' si no existe
torch.save(model.state_dict(), model_save_path)
print(f"Modelo guardado como {model_save_path}")