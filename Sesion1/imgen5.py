import cv2
import numpy as np

ancho = alto = 700

img = np.zeros((alto, ancho, 3), np.uint8)

for x in range(ancho):
    for y in range(alto):
        if x%50 == 0 and y%50 == 0:
            img[y, x] = 255

cv2.imshow("Imagen con puntos", img)
cv2.waitKey(0)
cv2.destroyAllWindows()