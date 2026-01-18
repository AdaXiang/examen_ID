import cv2 

img1 = cv2.imread('lapiz1.jpeg', 0)
img2 = cv2.imread('lapiz2.jpeg', 0)

cv2.imshow('Wikipedia0', img1)
cv2.waitKey(0)
cv2.imshow('Wikipedia3', img2)
cv2.waitKey(0)

img = cv2.subtract(img1, img2)

cv2.imshow('Sustraccion imagenes 1', img)
cv2.waitKey(0)

#--------------------------------------------------------
# img3 = cv2.imread('../imagenes/Mejostilla.jpeg', 1)
# img4 = cv2.imread('../imagenes/PaseoAlto.jpeg', 1)

# cv2.imshow('Mejostilla', img3)
# cv2.waitKey(0)
# cv2.imshow('PaseoAlto', img4)
# cv2.waitKey(0)

# imag = cv2.subtract(img3, img4)

# cv2.imshow('Sustraccion imagenes 2', imag)

# key = cv2.waitKey(0)
#--------------------------------------------------------
# img5 = cv2.imread('../imagenes/lapiz3.jpeg', 1)
# img6 = cv2.imread('../imagenes/lapiz4.jpeg', 1)

# cv2.imshow('lapiz3', img5)
# cv2.waitKey(0)
# cv2.imshow('lapiz4', img6)
# cv2.waitKey(0)

# imag2 = cv2.subtract(img5, img6)

# cv2.imshow('Sustraccion imagenes 3', imag2)

# key = cv2.waitKey(0)

# cv2.destroyAllWindows()