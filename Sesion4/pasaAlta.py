import cv2

# Cargar la imagen en escala de grises
#img = cv2.imread('../imagenes/motos.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('./panda.jpg', 0)

img_paso_bajo = cv2.blur(img, (10, 10))
img_paso_alto = cv2.subtract(img, img_paso_bajo)
cv2.imshow('Imagen filtrada', img_paso_alto)
#Sobel, Scharr, Laplacian, etc.
#ImagenPasaAlta = ImgOriginal - ImagenPasaBaja

cv2.imshow('Imagen original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

