import numpy as np
import cv2

color = (150, 120, 120)
grosor = 2
borrado = False

img = np.zeros((600, 600, 3), np.uint8)
img[:] = (255, 255, 255)
cv2.imshow('Pizarra',img)

def pinta(event,x,y,flags,param):
    global x_prev,y_prev, color, grosor

    if event == cv2.EVENT_LBUTTONDOWN:
        x_prev,y_prev = x,y

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.line(img,(x_prev,y_prev),(x,y),color, grosor)
        x_prev,y_prev = x,y

    cv2.imshow('Pizarra',img)

cv2.setMouseCallback('Pizarra',pinta)

while True:
    key = cv2.waitKey(100)

    #Tecla ENTER
    if key == 13:
        borrado = not(borrado)
        if borrado :
            color = (255, 255, 255)
            grosor = 8

        else :
            color = (0, 0, 255)
            grosor = 2

    #Tecla ESC ó 'q'
    elif key == 27 or key == ord('q'): break

cv2.destroyAllWindows()