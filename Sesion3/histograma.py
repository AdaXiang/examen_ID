
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen en escala de grises

#img = cv2.imread('../imagenes/motos.jpeg', cv2.IMREAD_GRAYSCALE)

img = cv2.imread('Mejostilla1.jpeg', 1)

# Calcular el histograma

hist = cv2.calcHist([img], [1], None, [256], [0, 256]) #[0] Azul si es en color
#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate])

# Mostrar la imagen
cv2.imshow('Imagen', img)
cv2.waitKey(0)  # Esperar a que se presione una tecla
cv2.destroyAllWindows()  # Cerrar la ventana de la imagen

# Configurar el gráfico del histograma
plt.figure(figsize=(10, 5))  # Tamaño de la figura
plt.plot(hist, color='black')  # Trazar el histograma
plt.xlim([0, 256])  # Limitar el eje x
plt.xlabel('Nivel de luz')
plt.ylabel('Número de píxeles')
plt.title('Histograma de la imagen')
plt.grid()  # Agregar una cuadrícula para mejor visualización
plt.show()  # Mostrar el gráfico