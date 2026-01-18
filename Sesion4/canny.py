#3fases: 
# 1) Filtrado de ruido gaussiano kernel 5x5

#2) Gradiente diferencia de intesidad píxeles adyacentes

#3) Histéresis: decidir si es borde o no, por encima del máximo sí

#Por debajo del mínimo no

#Los del medio clasificados por conectividad
import cv2


img = cv2.imread('./figuras_geometricas.png', 0)

img_canny = cv2.Canny(img, 100, 200) #UmbralInferior y UmbralSuperior

cv2.imshow('Imagen filtrada', img_canny)

cv2.imshow('Imagen original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()