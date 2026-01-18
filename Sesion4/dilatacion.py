import cv2
import numpy as np

# Elemento estructural
kernel = np.ones((8,8),np.uint8)

img = cv2.imread('./Invalidos_ruido.jpg', 0)

# El kernel es una matriz que define la vecindad para la operación de dilatación.
# En este caso, se utiliza un kernel de 2x2 lleno de unos.
# Se borran los puntos.
img_dilatada = cv2.dilate(img,kernel)

cv2.imshow('Imagen filtrada', img_dilatada)

cv2.imshow('Imagen original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()