import cv2
img = cv2.imread('./figuras_geometricas.png', 1)

cv2.imshow ('Figuras Geométricas', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_azul, img_verde, img_roja = cv2.split(img) #divir en colores primarios

# Para unirlas:
# merge(img_azul, img_verde, img_roja)
cv2.imshow('Azul', img_azul)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Verde', img_verde)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Roja', img_roja) 
cv2.waitKey(0)
cv2.destroyAllWindows()