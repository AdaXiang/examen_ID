#Diccionarios: colecciones de pares clave-valor
#--------------------------------------------------

# Crear un diccionario con contactos que contienen múltiples campos
contactos = {
    'Juan': {
        'email': 'juan@example.com',
        'telefono': '123-456-7890',
        'direccion': 'Calle 1, Ciudad A'

    },
    'Ana': {
        'email': 'ana@example.com',
        'telefono': '234-567-8901',
        'direccion': 'Calle 2, Ciudad B'

    },
    'Luis': {
        'email': 'luis@example.com',
        'telefono': '345-678-9012',
        'direccion': 'Calle 3, Ciudad C'
    }
}

# Imprimir el diccionario original
print("Diccionario original:", contactos)

# Acceder a la información de Ana
print("Email de Ana:", contactos['Ana']['email'])
print("Teléfono de Ana:", contactos['Ana']['telefono'])

# Modificar el número de teléfono de Luis
contactos['Luis']['telefono'] = '555-555-5555'
print("Diccionario después de modificar el teléfono de Luis:", contactos)

# Agregar un nuevo contacto con más campos
contactos['María'] = {
    'email': 'maria@example.com',
    'telefono': '456-789-0123',
    'direccion': 'Calle 4, Ciudad D'
}
print("Diccionario después de agregar a María:", contactos)

# Eliminar un contacto
del contactos['Juan']
print("Diccionario después de eliminar a Juan:", contactos)

# Comprobar si una clave está en el diccionario
print("¿Está 'Ana' en el diccionario?", 'Ana' in contactos)

# Recorrer el diccionario e imprimir cada contacto
print("Lista de contactos:")
for nombre, info in contactos.items():
    print(f"{nombre}: Email: {info['email']}, Teléfono: {info['telefono']}, Dirección: {info['direccion']}")

# Imprimir la longitud del diccionario
print("Número de contactos en el diccionario:", len(contactos))