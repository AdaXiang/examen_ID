
import cv2 

img1 = cv2.imread('panda.jpg', 1)
imgRoi = cv2.imread('seat.jpg', 1)

alto, ancho, _ = imgRoi.shape
roi1 = img1[0:alto, 0:ancho]

roi1 = cv2.bitwise_and(imgRoi, roi1)
img1[0:alto, 0:ancho ] = roi1

cv2.imshow('Composicion imagenes', img1)
key = cv2.waitKey(0)

#--------------------------------------------------------

img2 = cv2.imread('panda.jpg', 1)

roi2 = img2[0:alto, 0:ancho]
roi2 = cv2.add(imgRoi, roi2) #suma
img2[0:alto, 0:ancho ] = roi2

cv2.imshow('Composicion imagenes', img2)
key = cv2.waitKey(0)

#--------------------------------------------------------

img3 = cv2.imread('panda.jpg', 1)

roi3 = img3[0:alto, 0:ancho]
roi3 = cv2.subtract(roi3, imgRoi) #resta
img3[0:alto, 0:ancho ] = roi3

cv2.imshow('Composicion imagenes', img3)
key = cv2.waitKey(0)

cv2.destroyAllWindows()