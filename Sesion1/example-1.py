print("Hola mundo")

#Listas
#--------------------------------------------------
lista1 = [1, 2]       # lista1 apunta a una lista con [1, 2]
lista2 = []           # lista2 apunta a una lista vacía []
lista2 = lista1.copy()# lista2 apunta a una COPIA independiente de lista1 → [1, 2]
#lista2 = lista1       # ahora lista2 deja de apuntar a la copia y vuelve a apuntar a la MISMA lista que lista1
lista2.append(3)      # modificamos esa lista añadiendo un 3

print("elementos de lista1: " + str(lista1))
print("elementos de lista2: " + str(lista2))

# Crear una lista
frutas = ['manzana', 'plátano', 'naranja', 'fresa']

# Imprimir la lista original
print("Lista original:", frutas)

# Acceder a elementos de la lista
print("Primer elemento:", frutas[0])  # manzana
print("Segundo elemento:", frutas[1])  # plátano

# Modificar un elemento de la lista
frutas[2] = 'kiwi'
print("Lista después de modificar el tercer elemento:", frutas)

# Agregar un elemento al final de la lista
frutas.append('mango')
print("Lista después de agregar 'mango':", frutas)

# Eliminar un elemento de la lista
frutas.remove('plátano')
print("Lista después de eliminar 'plátano':", frutas)

# Recorrer la lista e imprimir cada fruta
print("Lista de frutas:")
for fruta in frutas:
    print(fruta)

# Imprimir la longitud de la lista
print("Número de frutas en la lista:", len(frutas))

