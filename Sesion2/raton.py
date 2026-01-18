import cv2
import numpy as np

# Crear una imagen blanca
img = np.ones((600, 600, 3), np.uint8)
img[:] = (255, 255, 255) # para todos los puntos de la imagen

# Configurar el color, grosor, fuente y escala del texto
color = (0, 0, 255)
color2 = (255, 0, 0)
grosor = 4
fuente = cv2.FONT_HERSHEY_SIMPLEX
escala = 1

# Mostrar la ventana inicial
cv2.imshow('Eventos raton', img)

# Definir la función para manejar los eventos del ratón
def eventos_raton(evento, x, y, flags, parametros):
    if evento == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(img, "Clic izquierdo", (x, y), fuente, escala, color, grosor)
    elif evento == cv2.EVENT_RBUTTONDOWN:
        cv2.putText(img, "Clic derecho", (x, y), fuente, escala, color2, grosor)
    cv2.imshow('Eventos raton', img)  # Actualizar la ventana con el texto

# Asignar la función de manejo de eventos a la ventana
cv2.setMouseCallback('Eventos raton', eventos_raton) # como el manejador de SO

# Mantener la ventana abierta hasta que se presione la tecla 'q'
while True: #no sirve en este programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas al salir del bucle
cv2.destroyAllWindows()