import cv2 

img1 = cv2.imread('./espacio.jpg', 1)
img2 = cv2.imread('./tierra.jpg', 1)

cv2.imshow('Espacio', img1)
cv2.waitKey(0) 
cv2.imshow('Tierra', img2)
cv2.waitKey(0) 

img = cv2.addWeighted(img1, 0.4, img2, 0.8, 0) # 0.4 y 0.8 son los pesos de cada imagen

#Último parametro es GAMMA q aumenta el nivel de luminosidad 0-255 # por el histograma
cv2.imshow('Composicion imagenes', img) 

key = cv2.waitKey(0)

img3 = cv2.imread('./Mejostilla1.jpeg', 1)
img4 = cv2.imread('./MejostillaArbol1.jpeg', 1)

cv2.imshow('Mejostilla', img3)
cv2.waitKey(0) 
cv2.imshow('Arbol', img4)
cv2.waitKey(0) 

imag = cv2.addWeighted(img3, 0.4, img4, 0.8, 0)

#Último parametro es GAMMA q aumenta el nivel de luminosidad 0-255
cv2.imshow('Composicion imagenes 2', imag)

key = cv2.waitKey(0)
cv2.destroyAllWindows()