import cv2
import numpy as np

img = cv2.imread('./panda.jpg', 1)
alto, ancho, canales = img.shape
color = (140, 30, 255)
incremento_radio = 20
grosor = 10

centro_x = ancho / 2
centro_y = alto / 2


for radio in range(0, int((alto, ancho)/2), incremento_radio):
   # Empieza desde el centro y va hacia afuera
   cv2.circle(img, (int(centro_x), int(centro_y)), radio, color, grosor)

cv2.imshow('Imagen con circulo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()