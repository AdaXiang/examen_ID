import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np

# Para protección en Windows (multiprocessing)
def main():
    
    """
    EXPLICACIÓN COMPLETA: ¿QUÉ ES UNA RED NEURONAL CONVOLUCIONAL (CNN)?
    
    Imagina que quieres enseñar a una computadora a distinguir gatos de perros en fotos.
    Una CNN es como un "detector de patrones" que aprende automáticamente qué buscar.
    
    ¿CÓMO FUNCIONA UNA CNN?
    1. CAPAS CONVOLUCIONALES: Buscan patrones locales como bordes, texturas, formas.
       - Usan "filtros" (kernels) que se deslizan sobre la imagen.
       - Ejemplo: Un filtro detecta líneas verticales, otro círculos.
    
    2. ACTIVACIÓN (ReLU): Hace que la red sea no-lineal, permitiendo aprender formas complejas.
       - Sin esto, sería solo una suma lineal (como una regresión).
    
    3. POOLING: Reduce el tamaño de la imagen, manteniendo lo importante.
       - Max Pooling: Toma el valor máximo en una ventana (conserva bordes fuertes).
       - Reduce ruido y acelera el procesamiento.
    
    4. BATCH NORMALIZATION: Estabiliza el entrenamiento normalizando activaciones.
       - Evita que capas profundas cambien mucho, acelera convergencia.
    
    5. DROPOUT: Apaga neuronas aleatoriamente durante entrenamiento.
       - Previene sobreajuste (memorizar datos en vez de generalizar).
    
    6. FULLY CONNECTED: Toma las características extraídas y clasifica.
       - Como una red normal: combina todo para decidir "gato" o "perro".
    
    ¿QUÉ ES EL ENTRENAMIENTO?
    - FORWARD PASS: Imagen entra, pasa por capas, sale predicción.
    - LOSS: Mide qué tan equivocada está la predicción.
    - BACKWARD PASS: Calcula cómo cambiar pesos para reducir error.
    - OPTIMIZER: Actualiza pesos usando gradientes (descenso de gradiente).
    
    EN ESTE CÓDIGO:
    - Usamos 4 bloques convolucionales para extraer características.
    - Data augmentation crea variaciones para robustez.
    - Entrenamos en batches, validamos en datos no vistos.
    - Scheduler reduce learning rate si se estanca.
    
    RESULTADO: Un modelo que clasifica imágenes con ~80%+ accuracy.
    """
    
    # --- 1. CONFIGURACIÓN BÁSICA ---
    # Una Red Neuronal Convolucional (CNN) es un tipo de red que procesa imágenes.
    # Funciona extrayendo características (bordes, texturas) con capas convolucionales,
    # reduciendo el tamaño con pooling, y clasificando con capas fully connected.
    # El entrenamiento ajusta los pesos para minimizar errores en predicciones.
    
    # Aumentamos a 40 épocas porque los datos son ahora "más difíciles" de aprender
    NUM_EPOCHS = 40  # Una época es un pase completo por todos los datos de entrenamiento
    BATCH_SIZE = 64   # Número de imágenes procesadas juntas (acelera y estabiliza el entrenamiento)
    LEARNING_RATE = 0.001  # Qué tan rápido aprende el modelo (bajo = lento pero estable)
    IMG_SIZE = 128  # Tamaño 128x128 (Balance perfecto velocidad/precisión)

    # Detección de dispositivo
    # Las GPUs aceleran el entrenamiento al procesar matrices en paralelo.
    # MPS es para Macs con GPU Apple, CUDA para NVIDIA, CPU como fallback.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # --- 2. TRANSFORMACIONES (DATA AUGMENTATION) ---
    # Las imágenes se preprocesan para que el modelo aprenda mejor.
    # Data Augmentation crea variaciones (giro, espejo) para evitar sobreajuste.
    # Normalización ajusta colores a rangos estándar (mejora estabilidad).
    
    # Aquí aplicamos la "Aumentación Agresiva" para romper el techo del 74%
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Cambia tamaño de imagen
        
        # >>>> CAMBIOS CLAVE ANTI-OVERFITTING <<<<
        transforms.RandomHorizontalFlip(p=0.5),           # Espejo horizontal (50% chance)
        transforms.RandomRotation(20),                    # Rotación hasta 20 grados
        transforms.RandomAffine(degrees=0, 
                                translate=(0.1, 0.1),     # Mueve la imagen un 10%
                                scale=(0.9, 1.1)),        # Zoom in/out aleatorio
        transforms.ColorJitter(brightness=0.2,            # Variación de luz/color
                               contrast=0.2, 
                               saturation=0.2),
        # >>>> FIN CAMBIOS <<<<
        
        transforms.ToTensor(),  # Convierte imagen a tensor (números para la red)
        transforms.Normalize([0.485, 0.456, 0.406],      # Normaliza colores (media)
                             [0.229, 0.224, 0.225])      # y desviación estándar
    ])

    # Validación y Test (Solo redimensionar, sin trucos)
    # No augmentamos val/test para evaluar el modelo "real"
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- 3. DATASETS Y DATALOADERS ---
    # Dataset: Colección de imágenes etiquetadas (gato/perro).
    # DataLoader: Carga datos en batches para entrenamiento eficiente.
    
    # Ajusta la ruta "data/train" si tu carpeta se llama diferente
    train_dataset = datasets.ImageFolder(root="data/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(root="data/validation", transform=data_transforms)
    test_dataset = datasets.ImageFolder(root="data/test", transform=data_transforms)
    
    class_names = train_dataset.classes  # ['cats', 'dogs']
    print(f"Classes: {class_names}")
    print(f"Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)}")

    # num_workers=2 acelera carga en CPUs múltiples (0 en Windows si hay errores)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 4. EL MODELO (DNNclass - 128x128) ---
    # Definimos la arquitectura de la CNN.
    # Las capas convolucionales detectan patrones (bordes, formas).
    # Pooling reduce tamaño y ruido.
    # BatchNorm estabiliza entrenamiento.
    # Dropout evita sobreajuste apagando neuronas aleatoriamente.
    # Fully Connected clasifican basado en características extraídas.
    
    # ARQUITECTURA DETALLADA:
    # Entrada: Imagen 128x128x3 (alto x ancho x canales RGB)
    # Después de 4 bloques: 8x8x128 (características comprimidas)
    # Flatten: Vector de 8192 números
    # FC layers: De 8192 -> 512 -> 2 (probabilidades gato/perro)
    
    class DNNclass(nn.Module):
        def __init__(self, num_classes):
            super(DNNclass, self).__init__()
            
            # BLOQUE 1: Detección de características básicas (bordes, colores)
            # Conv2d: Aplica filtros 3x3, padding=1 mantiene tamaño
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 128x128x3 -> 128x128x32
            self.bn1 = nn.BatchNorm2d(32)  # Normaliza para estabilidad
            self.pool = nn.MaxPool2d(2, 2)  # 128x128x32 -> 64x64x32 (reduce tamaño)
            
            # BLOQUE 2: Patrones más complejos (formas simples)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64x32 -> 64x64x64
            self.bn2 = nn.BatchNorm2d(64)
            # pool: 64x64x64 -> 32x32x64
            
            # BLOQUE 3: Características avanzadas (partes del animal)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32x64 -> 32x32x128
            self.bn3 = nn.BatchNorm2d(128)
            # pool: 32x32x128 -> 16x16x128
            
            # BLOQUE 4: Detalles finos y composición
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 16x16x128 -> 16x16x128
            self.bn4 = nn.BatchNorm2d(128)
            # pool: 16x16x128 -> 8x8x128
            
            # FLATTEN: Convierte la "imagen" 8x8x128 en vector 1x8192
            self.flatten_size = 128 * 8 * 8  # 8192
            
            # CAPAS FULLY CONNECTED: Clasificación final
            self.fc1 = nn.Linear(self.flatten_size, 512)  # 8192 -> 512 (reduce dimensionalidad)
            self.dropout = nn.Dropout(0.5)  # Apaga 50% neuronas aleatoriamente
            self.fc2 = nn.Linear(512, num_classes)  # 512 -> 2 (gato=0, perro=1)

        def forward(self, x):
            # PASE HACIA ADELANTE: Cómo procesa la imagen
            
            # Bloque 1: Extracción básica
            x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BatchNorm -> ReLU -> Pool
            
            # Bloque 2: Formas
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            
            # Bloque 3: Partes
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            
            # Bloque 4: Composición
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            
            # Aplanar para fully connected
            x = x.view(x.size(0), -1)  # De [batch, 128, 8, 8] a [batch, 8192]
            
            # Clasificación
            x = F.relu(self.fc1(x))  # FC -> ReLU
            x = self.dropout(x)      # Dropout (solo en train)
            x = self.fc2(x)          # Salida: logits (antes de softmax)
            return x

    model = DNNclass(num_classes=len(class_names)).to(device)

    # Loss y Optimizer
    # Loss: Mide error entre predicción y realidad (CrossEntropy para clasificación)
    criterion = nn.CrossEntropyLoss()
    # Optimizer: Actualiza pesos para reducir loss (Adam combina momentum y RMSProp)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Scheduler: Reduce LR si se estanca 5 épocas (más paciencia)
    # Evita que el modelo deje de aprender si el loss no baja
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=5)

    # --- 5. FUNCIONES DE ENTRENAMIENTO ---
    # Entrenamiento: Ajusta pesos usando datos. Forward: Predice, calcula loss.
    # Backward: Gradientes indican cómo cambiar pesos. Optimizer actualiza.
    
    def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
        model.train()  # Modo entrenamiento (usa dropout, batchnorm)
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Limpia gradientes previos
            outputs = model(images)  # Forward pass: Predicciones
            loss = criterion(outputs, labels)  # Calcula error
            loss.backward()  # Backward pass: Gradientes
            optimizer.step()  # Actualiza pesos
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)  # Clase con mayor probabilidad
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(acc=correct/total, loss=loss.item())
            
        return running_loss / total, correct / total

    def evaluate(model, loader, criterion, device):
        model.eval()  # Modo evaluación (no dropout, batchnorm usa stats guardadas)
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # No calcula gradientes (más rápido)
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return running_loss / total, correct / total

    def get_predictions(model, loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # Índice de clase con mayor score
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_preds, all_labels

    # --- 6. BUCLE PRINCIPAL ---
    # Entrenamiento por épocas: Repite train/val hasta converger.
    # Train ajusta pesos, Val mide generalización (sin ajustar).
    print("Iniciando entrenamiento...")
    
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        # Entrena con datos de train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        # Evalúa en val (datos no vistos en train)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Guarda métricas para gráficos
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Scheduler ajusta LR basado en val_loss
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        
        # Guarda modelo si mejora en val (evita sobreajuste)
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")

    print(f"\nMejor Accuracy en Validación: {best_acc:.4f}")

    # Evaluación Final
    # Test set: Datos nunca vistos, mide rendimiento real.
    print("Evaluando en Test Set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy Final: {test_acc:.4f}")
    
    # Guardar modelo final (último estado)
    torch.save(model.state_dict(), "models/cats_dogs_cnn.pth")
    print("Modelo final guardado en models/cats_dogs_cnn.pth")
    
    # Matriz de Confusión: Muestra errores por clase (ej. cuántos gatos clasificados como perros)
    test_preds, test_labels = get_predictions(model, test_loader, device)
    cm = confusion_matrix(test_labels, test_preds)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión (Acc: {test_acc:.2%})')
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.show()

# Bloque seguro para Windows
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
