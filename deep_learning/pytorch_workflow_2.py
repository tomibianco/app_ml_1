import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import io


class ImageClassifier:
    def __init__(self, data_dir, num_classes=3, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Definir transformaciones para las imágenes
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        # Transformación para predicción
        self.predict_transform = self.data_transforms['val']
        
    def load_data(self):
        # Cargar datasets
        self.image_datasets = {
            x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
            for x in ['train', 'val']
        }
        
        # Crear dataloaders
        self.dataloaders = {
            x: DataLoader(self.image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
            for x in ['train', 'val']
        }
        
        self.class_names = self.image_datasets['train'].classes
        
    def setup_model(self):
        # Cargar ResNet-50 preentrenado
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Congelar todos los parámetros
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modificar la última capa fully connected
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        # Mover modelo a GPU si está disponible
        self.model = self.model.to(self.device)
        
        # Definir criterio y optimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        
    def train_model(self, num_epochs=10):
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Cada época tiene una fase de entrenamiento y validación
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                
                # Iterar sobre los datos
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Poner gradientes a cero
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward pass + optimize solo en fase de entrenamiento
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(self.image_datasets[phase])
                epoch_acc = running_corrects.double() / len(self.image_datasets[phase])
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Guardar el mejor modelo
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.save_model('best_model.pth')
                    
        print(f'Best val Acc: {best_acc:4f}')
        
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_names': self.class_names
        }, path)
        
    def load_saved_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.class_names = checkpoint['class_names']
        
    def predict(self, image_path):
        # Cargar y preparar la imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.predict_transform(image).unsqueeze(0).to(self.device)
        
        # Hacer predicción
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, preds = torch.max(outputs, 1)
            
        return self.class_names[preds[0]]

# API con FastAPI
app = FastAPI()

# Instancia global del modelo
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = ImageClassifier(data_dir="ruta/a/tus/datos")
    model.setup_model()
    model.load_saved_model("best_model.pth")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Guardar temporalmente la imagen
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Hacer predicción
    prediction = model.predict(temp_path)
    
    # Eliminar archivo temporal
    os.remove(temp_path)
    
    return {"prediction": prediction}