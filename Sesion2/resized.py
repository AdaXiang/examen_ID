import cv2

img = cv2.imread('./panda.jpg', 1)

alto, ancho, _ = img.shape
escala = 1.3
ancho_escalado, alto_escalado = int(ancho*escala), int(alto*escala)

imagen_escalada = cv2.resize(img, (ancho_escalado, alto_escalado))

#3r argumento opcional INTERPOLACIÓN => cuando se aumenta la foto se deben añadir píxeles y se reduce eliminarlos
#INTER_AREA al reducir el tamaño
#INTER_LINEAR para hacer zoom
#INTER_CUBIC más eficiente pero más lento

cv2.imshow('Imagen original', img)
cv2.imshow('Imagen escalada', imagen_escalada)
cv2.waitKey(0)
cv2.destroyAllWindows()

