import cv2
img = cv2.imread('./panda.jpg', 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convertir a escala de grises
edges = cv2.Canny(gray, 100, 200) #detectar bordes
cv2.imshow ('Figuras Geométricas', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
