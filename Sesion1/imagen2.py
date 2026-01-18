import cv2
import numpy

ancho = alto = 300

#img = numpy.ones((alto, ancho, 3), numpy.uint8) * 255 
img = numpy.zeros((alto, ancho, 3), numpy.uint8)
img[:] = (100, 255, 255)

cv2.imshow("Imagen a color", img) #muestra la imagen
cv2.waitKey(0) #espera a que le de a una tecla
cv2.destroyAllWindows()