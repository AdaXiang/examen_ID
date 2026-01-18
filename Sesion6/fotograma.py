
import cv2



#Se crea el objeto que representa la fuente de video

camara = cv2.VideoCapture(0)

# Se obtiene el código FourCC del codec DIVX

#compatible con QuickTime Player

#fourcc = cv2.VideoWriter_fourcc(*'DIVX')

#fourcc = cv2.VideoWriter_fourcc(*'HEVC')

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')

fourcc = cv2.VideoWriter_fourcc(*'H264')

# Se obtiene el código FourCC del codec 'mp4v' para formato MP4

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')



#Se crea el objeto VideoWriter 

width = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))



video = cv2.VideoWriter('../videos/video.mp4', fourcc, 20, (width, height))

#Si no se ha podido acceer a la fuente de video se sale del programa

if not camara.isOpened():

    print("No es posible abrir la cámara")

    exit()

while True:

    # Se captura la imagen frame a frame

    ret, frame = camara.read()

    # Si la captura no se ha tomado correctamente se sale del bucle

    if not ret:

        print("No es posible obtener la imagen")

        break

    # El frame se muestra en pantalla

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) == ord('q'):

        break

    elif cv2.waitKey(1) == ord('s'):

        print('Fotograma guardado')

        cv2.imwrite('../imagenes/fotograma.jpg', frame)

# Se liberan los recursos utilizados y se cierra la ventana