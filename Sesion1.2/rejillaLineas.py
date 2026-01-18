import cv2
import numpy as np

img = cv2.imread('./panda.jpg', 1)#El uno es para leer la imagen a color
#img = cv2.imread('./panda.jpg', 0)#El cero es para leer la imagen en blanco y negro

alto, ancho, _ = img.shape
color = (140, 30, 255)
grosor = 2
cuadricula = 80

# Dibujar líneas verticales y horizontales para crear una cuadrícula
for x in range(0, ancho+1, cuadricula):
    img = cv2.line(img, (x, 0), (x, alto), color, grosor)
for y in range(0, alto+1, cuadricula):
    img = cv2.line(img, (0, y), (ancho, y), color, grosor)

cv2.imshow('Imagen con cuadrícula', img)
cv2.waitKey(0)
cv2.destroyAllWindows()