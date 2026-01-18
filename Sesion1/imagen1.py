import cv2
import numpy as np

ancho = alto = 300
img = np.zeros((alto, ancho), np.uint8) #crea una imagen negra
img = np.ones((alto, ancho), np.uint8)*255 #crea una imagen blanca

cv2.imshow("Imagen Negra", img) #muestra la imagen
cv2.waitKey(0) #espera a que le de a una tecla
cv2.destroyAllWindows()