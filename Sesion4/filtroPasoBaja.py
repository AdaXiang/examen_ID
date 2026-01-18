import cv2

# Cargar la imagen en escala de grises 
#img = cv2.imread('../imagenes/motos.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('./panda.jpg', 0)

img_suavizada = cv2.blur(img, (30, 30)) # Difumina la imagen
#GaussianBlur()
#medianBlur(
#bilateralFilter()

cv2.imshow('Imagen filtrada', img_suavizada)

cv2.imshow('Imagen original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()