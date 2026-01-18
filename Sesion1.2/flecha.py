import cv2
import numpy as np

img = cv2.imread('./panda.jpg', 1)

# Origen y destino de las flechas, color y grosor
cv2.arrowedLine(img,  (100, 1500), (200, 300), (25, 140, 0), 4)
cv2.arrowedLine(img, (200, 50), (50, 200), (0, 25, 140), 2)  

cv2.imshow('Imagen con cuadrícula', img)
cv2.waitKey(0)
cv2.destroyAllWindows()