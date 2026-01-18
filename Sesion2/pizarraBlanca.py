import numpy as np
import cv2

color = (140, 80, 120)
grosor = 2

img = np.zeros((600, 600, 3), np.uint8)
img[:] = (255, 255, 255) # ventana blanca
cv2.imshow('Pizarra',img)

def pinta(event,x,y,flags,param):
    global x_prev,y_prev
    if event == cv2.EVENT_LBUTTONDOWN:
        x_prev,y_prev = x,y

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.line(img,(x_prev,y_prev),(x,y),color, grosor)
        x_prev,y_prev = x,y

    cv2.imshow('Pizarra',img)

cv2.setMouseCallback('Pizarra',pinta)

# Mantener la ventana abierta hasta que se presione la tecla 'q'
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas al salir del bucle
cv2.destroyAllWindows()