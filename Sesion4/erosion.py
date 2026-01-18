#Elimina ruido blanco && separa objetos que están conectados
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

img = cv2.imread('./agujeros.jpg', 0)
img_erosionada = cv2.erode(img,kernel)

cv2.imshow('Imagen filtrada', img_erosionada)

cv2.imshow('Imagen original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#morphologyEx(imagen, cv.MORPH_OPEN, área estructurante)
#APERTURA = erosión -> dilatación
#CIERRE ó CLAUSURA = dilatación -> erosión