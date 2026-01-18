import cv2
import numpy as np

img = cv2.imread('./panda.jpg', 1)

color = (140, 30, 255)
grosor = 2
rectangulo_x1, rectangulo_x2 = 80, 500
rectangulo_y1, rectangulo_y2 = 100, 500
# Dibujar un rectángulo
cv2.rectangle(img, (rectangulo_x1, rectangulo_y1), (rectangulo_x2, rectangulo_y2), color, grosor)


cv2.imshow('Imagen con cuadrícula', img)
cv2.waitKey(0)
cv2.destroyAllWindows()