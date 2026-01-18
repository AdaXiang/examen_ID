#Tuplas: es fija, no se puede modificar los elementos
#--------------------------------------------------

# Crear una tupla
frutas = ('manzana', 'plátano', 'naranja', 'fresa')

# Imprimir la tupla original
print("Tupla original:", frutas)

# Acceder a elementos de la tupla
print("Primer elemento:", frutas[0])  # manzana
print("Segundo elemento:", frutas[1])  # plátano

# Intentar modificar un elemento (esto generará un error)
try:
    frutas[2] = 'kiwi'  # Esto causará un error
except TypeError as e:
    print("Error:", e)

# Crear una nueva tupla con un elemento adicional
frutas_nuevas = frutas + ('kiwi',)
print("Tupla nueva después de agregar 'kiwi':", frutas_nuevas)

# Recorrer la tupla e imprimir cada fruta
print("Lista de frutas en la tupla:")
for fruta in frutas:
    print(fruta)

# Imprimir la longitud de la tupla
print("Número de frutas en la tupla:", len(frutas))



