import cv2
img = cv2.imread('./panda.jpg', 1)

tamanio = img.size
alto, ancho, canales = img.shape
tipo = img.dtype

print("Tamaño: " + str(tamanio) + " bytes")
print("Ancho: " + str(ancho) + " píxeles")
print("Alto: " + str(alto) + " píxeles")
print("Nº canales: " + str(canales))
print("Tipo: " + str(tipo))

cv2.imshow('MOTOS', img)
cv2.waitKey(0)
cv2.destroyAllWindows()