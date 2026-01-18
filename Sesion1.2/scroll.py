import cv2 

img = cv2.imread('./panda.jpg', 1)

img_original = img.copy()
color = (0, 0, 255)
grosor = 4
fuente = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
escala = 2
alto_imagen, ancho_imagen, canales = img.shape

def actualizar_imagen(escala):
    img = img_original.copy()
    posicion_x, posicion_y = centrarImagen(escala)
    grosor = escala
    cv2.putText(img, "3 ruedas", (posicion_x, posicion_y), fuente, escala, color, grosor)
    cv2.imshow('Motos', img)

def centrarImagen(escala):
    (ancho_texto, alto_texto), _ = cv2.getTextSize("3 ruedas", fuente, escala, grosor)
    posicion_x = int((ancho_imagen - ancho_texto) / 2)
    posicion_y = int((alto_imagen + alto_texto) / 2)
    return posicion_x, posicion_y

# Crear ventana antes de usar el trackbar
cv2.namedWindow('Motos')
# moveWindow (name, x, y)
# resizeWindow (name, ancho, alto)
# destroyWindow (name)
# destroyAllWindows()

#Mostrar imagen inicial
#posicion_x, posicion_y = centrarImagen(escala)
#cv2.putText(img, "3 ruedas", (posicion_x, posicion_y), fuente, escala, color, grosor)
#cv2.imshow('motos', img)

#Crear el trackbar
cv2.createTrackbar('Escala texto', 'Motos', 10, 30, actualizar_imagen)

# Bucle para actualizar la imagen y el trackbar en tiempo real
while True:
    # Obtener la posición actual del trackbar
    escala = cv2.getTrackbarPos('Escala texto', 'Motos')

    # Actualizar la imagen según la escala
    actualizar_imagen(escala)

    # Salir del bucle si se presiona la tecla 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:  # Código ASCII para 'ESC'
        break

cv2.destroyAllWindows()