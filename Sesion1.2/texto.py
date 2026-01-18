
import cv2 

img = cv2.imread('./panda.jpg', 1)
color = (170, 70, 180)
grosor = 4
fuente = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
escala = 2

cv2.arrowedLine(img, (100, 550), (190, 520), color, grosor)
cv2.putText(img, "panda", (200, 525), fuente, escala, color, grosor)

cv2.imshow('motos', img)
cv2.waitKey(0)
cv2.destroyAllWindows()