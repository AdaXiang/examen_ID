import torch
from torchvision import transforms
from PIL import Image
import sys
import torch.nn as nn
import torch.nn.functional as F

# =========================
# MISMA CNN DEL ENTRENAMIENTO (DNNclass)
# =========================
class DNNclass(nn.Module):
    def __init__(self, num_classes):
        super(DNNclass, self).__init__()
        
        # Bloque 1
        # Entrada: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # -> 64 x 64
        
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pool -> 32 x 32
        
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # pool -> 16 x 16
        
        # Bloque 4 (Añadido para dar más potencia)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # pool -> 8 x 8
        
        # CAPA FLATTEN + DENSE (Estilo clásico Kaggle)
        # La imagen final es 128 filtros x 8 x 8 píxeles
        self.flatten_size = 128 * 8 * 8 
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5) # Importante para no memorizar
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Bloque 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Bloque 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Bloque 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Bloque 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Aplanar (Flatten)
        x = x.view(x.size(0), -1) 
        
        # Capas Densas
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =========================
# CARGAR EL MODELO
# =========================
MODEL_PATH = "models/best_model.pth"

model = DNNclass(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

class_names = ['cats', 'dogs']

# =========================
# TRANSFORMACIONES IGUALES (128x128)
# =========================
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# FUNCIÓN DE PREDICCIÓN
# =========================
def predict_image(path):
    img = Image.open(path).convert("RGB")
    img_t = data_transforms(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    print(f"\nImage: {path}")
    print(f"Prediction: {class_names[pred.item()]}\n")

# =========================
# MODO COMANDO
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
