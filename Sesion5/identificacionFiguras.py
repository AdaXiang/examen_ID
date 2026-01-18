#CONTORNOS DE APROXIMACIÓN
#approxPolyDP (contorno, error, tipo)
#Algoritmo de Douglas-Peucker

import cv2

umbral = 200
porcentaje_error = 0.01
#porcentaje_error = 0.15

fuente = cv2.FONT_HERSHEY_SIMPLEX
color = (0,0,0)
grosor = 2
escala = 1
texto = ""

img = cv2.imread('./figuras_geometricas5.jpg')

img_byn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_umbral = cv2.threshold(img_byn, umbral, 255, cv2.THRESH_BINARY_INV)

contornos, _ = cv2.findContours(img_umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contorno in contornos:
    x,y,ancho,alto = cv2.boundingRect(contorno)

    margen_error = porcentaje_error*cv2.arcLength(contorno, True )
    contorno_aprox = cv2.approxPolyDP(contorno, margen_error, True ) #Aproximación del contorno a una figura
    cv2.drawContours(img, [contorno_aprox], 0, color, grosor)

    if len(contorno_aprox) == 3: texto = "TRIANGULO"   # Si tiene 3 vértices es un triángulo
    elif len(contorno_aprox) == 4:texto = "CUADRADO"   # Si tiene 4 vértices es un cuadrado o rectángulo
    elif len(contorno_aprox) == 5: texto = "PENTAGONO" # Si tiene 5 vértices es un pentágono

    (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, fuente, escala, grosor)
    # Centrar el texto
    posicion_x = int(x + (ancho - ancho_texto) / 2)
    posicion_y = int(y + (alto / 2 + alto_texto / 2))

    cv2.putText(img, texto, (posicion_x, posicion_y), fuente, escala, color, grosor)

cv2.imshow('Figuras geometricas', img)

cv2.waitKey(0)
cv2.destroyAllWindows()