import cv2

img = cv2.imread('./panda.jpg', 1)
img_original = img.copy()

cv2.imshow('Motos', img)

def color(event,x,y,flags,param):
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        color = img[y, x].tolist() # convierte a lista
        cv2.circle(img, (x, y), 40,color, -1) #círculo relleno

    elif event == cv2.EVENT_LBUTTONUP: #cuando se deja de pulsar se borra el círculo
        img = img_original.copy()
    cv2.imshow('Motos',img)

cv2.setMouseCallback('Motos',color)
key = cv2.waitKey(0)
cv2.destroyAllWindows()