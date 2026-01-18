#Conjuntos: colección desordenada de elementos únicos, es decir, no permite duplicados
#--------------------------------------------------
# Crear un conjunto
frutas = {'manzana', 'plátano', 'naranja', 'fresa'}

# Imprimir el conjunto original
print("Conjunto original:", frutas)

# Agregar un elemento al conjunto
frutas.add('kiwi')
print("Conjunto después de agregar 'kiwi':", frutas)

# Intentar agregar un elemento duplicado (no se añadirá)
frutas.add('manzana')
print("Conjunto después de intentar agregar 'manzana' de nuevo:", frutas)

# Eliminar un elemento del conjunto
frutas.remove('plátano')
print("Conjunto después de eliminar 'plátano':", frutas)

# Comprobar si un elemento está en el conjunto
print("¿Está 'naranja' en el conjunto?", 'naranja' in frutas)

# Crear otro conjunto
citrus = {'naranja', 'limón', 'pomelo'}

# Realizar la unión de conjuntos
union = frutas.union(citrus)
print("Unión de conjuntos:", union)

# Realizar la intersección de conjuntos
interseccion = frutas.intersection(citrus)
print("Intersección de conjuntos:", interseccion)

# Imprimir la longitud del conjunto
print("Número de frutas en el conjunto:", len(frutas))
