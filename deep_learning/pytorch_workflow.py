import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# Transformaciones: redimensionar, convertir a tensor y normalizar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar los datos de entrenamiento y validación
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)








# Cargar un modelo preentrenado (ResNet50)
model = models.resnet50(pretrained=True)

# Congelar las capas del modelo base para no entrenarlas
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la última capa para la clasificación de tus clases (en este caso 2 clases)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # El número de clases es el tamaño de tu dataset

# Asegúrate de que las capas modificadas puedan ser entrenadas
for param in model.fc.parameters():
    param.requires_grad = True







# Configurar el dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definir el criterio de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)




# Entrenar el modelo (considerar mejorar con Data Augmentation)
num_epochs = 10  # Número de épocas de entrenamiento

for epoch in range(num_epochs):
    model.train()  # Poner el modelo en modo de entrenamiento
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Limpiar los gradientes

        outputs = model(inputs)  # Obtener las predicciones
        loss = criterion(outputs, labels)  # Calcular la pérdida
        loss.backward()  # Retropropagación
        optimizer.step()  # Actualizar los pesos

        running_loss += loss.item()

        # Calcular precisión
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")





model.eval()  # Poner el modelo en modo de evaluación
correct = 0
total = 0

with torch.no_grad():  # No calcular gradientes durante la evaluación
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = 100 * correct / total
print(f"Validation Accuracy: {val_acc:.2f}%")





torch.save(model.state_dict(), 'mining_lithium_model.pth')





model.load_state_dict(torch.load('mining_lithium_model.pth'))
model.eval()

# Cargar y transformar una nueva imagen
from PIL import Image

image = Image.open('path_to_new_image.jpg')
image = transform(image).unsqueeze(0).to(device)

# Hacer la predicción
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print(f'Predicted Class: {train_dataset.classes[predicted]}')