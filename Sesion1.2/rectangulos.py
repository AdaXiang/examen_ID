import cv2
import numpy as np

img = cv2.imread('./panda.jpg', 1)

alto, ancho, canales = img.shape
color = (140, 30, 255)
grosor = 2
separacion = 10
numerp_cuadros = 10

centro_x = ancho / 2
centro_y = alto / 2

for lado in range(0, 100, separacion):
   # Empieza desde el centro y va hacia afuera
   x1 = centro_x - int(lado*ancho/200)
   y1 = centro_y - int(lado*ancho/200)
   x2 = centro_x + int(lado*ancho/200)
   y2 = centro_y + int(lado*ancho/200)

   cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)

cv2.imshow('Imagen con cuadrícula', img)
cv2.waitKey(0)
cv2.destroyAllWindows()