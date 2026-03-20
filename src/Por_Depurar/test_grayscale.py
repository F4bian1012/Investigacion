import os
from PIL import Image

def convertir_a_grises_y_medir(ruta_entrada, ruta_salida):
    try:
        # 1. Verificar si el archivo existe
        if not os.path.exists(ruta_entrada):
            print(f"Error: El archivo '{ruta_entrada}' no existe.")
            return

        # 2. Obtener tamaño original en Bytes y convertir a kB
        # (1 kB = 1024 Bytes)
        tamano_bytes_antes = os.path.getsize(ruta_entrada)
        tamano_kb_antes = tamano_bytes_antes / 1024

        # 3. Abrir la imagen y convertirla
        # 'L' es el modo para escala de grises (Luminancia)
        with Image.open(ruta_entrada) as img:
            imagen_gris = img.convert('L')
            
            # Guardar la nueva imagen
            # Puedes ajustar la calidad si es JPEG (ej: quality=85) para reducir más peso
            imagen_gris.save(ruta_salida)

        # 4. Obtener tamaño final
        tamano_bytes_despues = os.path.getsize(ruta_salida)
        tamano_kb_despues = tamano_bytes_despues / 1024

        # 5. Calcular la diferencia
        diferencia = tamano_kb_antes - tamano_kb_despues
        porcentaje = (diferencia / tamano_kb_antes) * 100

        # 6. Mostrar resultados
        print("-" * 40)
        print(f"RESULTADOS DE LA CONVERSIÓN")
        print("-" * 40)
        print(f"Archivo original: {ruta_entrada}")
        print(f"Tamaño ANTES:     {tamano_kb_antes:.2f} kB")
        print("-" * 40)
        print(f"Archivo generado: {ruta_salida}")
        print(f"Tamaño DESPUÉS:   {tamano_kb_despues:.2f} kB")
        print("-" * 40)
        
        if diferencia > 0:
            print(f"Has ahorrado:     {diferencia:.2f} kB ({porcentaje:.2f}%)")
        else:
            print(f"El tamaño aumentó: {abs(diferencia):.2f} kB")
            print("(Esto puede pasar si cambias de formato comprimido JPG a PNG)")

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# --- Ejemplo de uso ---
# Asegúrate de cambiar 'mi_imagen.jpg' por una imagen real en tu carpeta
input_file = "data/raw/imagenes/Motospl10000.jpg"
output_file = "data/processed/grayscale/Motospl10000-grayscale.jpg"

# Nota: Para probarlo, crea o descarga una imagen llamada 'imagen_original.jpg' 
# en la misma carpeta donde guardes este script.
convertir_a_grises_y_medir(input_file, output_file)