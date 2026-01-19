
# FICHERO PARA GENERAR EL MODELO ...

# --------------------

#pip3 install torch torchvision matplotlib

#python3 perros_gatos.py



#Para usar la GPU de MAC m3:

if torch.backends.mps.is_available():

    device = torch.device("mps")

    print("Using device: MPS (Apple GPU)")

elif torch.cuda.is_available():

    device = torch.device("cuda")

    print("Using device: CUDA")

else:

    device = torch.device("cpu")

    print("Using device: CPU")



Quitando:

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print("Using device:", device)





# ¿Qué necesitas para usar el modelo?

# - El archivo .pth

# - La clase SimpleCNN que usaste al entrenarlo

# - Un script de predicción: 

#    python3 predict.py ejemploDivertido/disfraz.jpg

# - Una imagen a clasificar