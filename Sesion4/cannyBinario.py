#3fases: 1) Filtrado de ruido gaussiano kernel 5x5

#2) Gradiente diferencia de intesidad píxeles adyacentes

#3) Histéresis: decidir si es borde o no, por encima del máximo sí

#Por debajo del mínimo no

#Los del medio clasificados por conectividad



import cv2

  

# Cargar la imagen en escala de grises

#img = cv2.imread('../imagenes/motos.jpeg', cv2.IMREAD_GRAYSCALE)

img = cv2.imread('../imagenes/motos.jpeg', 0)



img_canny = cv2.Canny(img, 100, 200)

_, img_inversa = cv2.threshold(img_canny, 0, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('Imagen filtrada', img_inversa)



cv2.waitKey(0)

cv2.destroyAllWindows()

()