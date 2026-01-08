import cv2
import os
import glob

# Paths
RAW_DIR = "data/raw/imagenes"
PROCESSED_DIR = "data/processed/grayscale"

def get_size_kb(path):
    """Devuelve el tamaño del archivo en kB"""
    if os.path.exists(path):
        return os.path.getsize(path) / 1024
    return 0

def process_images():
    """
    Convierte imágenes a escala de grises controlando la compresión
    para evitar que aumente el tamaño del archivo.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    
    print(f"🔍 Buscando imágenes en {RAW_DIR}...")
    for ext in extensions:
        # Usamos recursive=True por si acaso, aunque en glob simple no es necesario
        files.extend(glob.glob(os.path.join(RAW_DIR, ext)))
        
    if not files:
        print("⚠️ No se encontraron imágenes.")
        return

    print(f"📸 Encontradas {len(files)} imágenes. Iniciando procesamiento...\n")
    
    processed_count = 0
    
    for file_path in files:
        try:
            # 1. Leer imagen
            img = cv2.imread(file_path)
            if img is None:
                print(f"❌ Error al cargar: {file_path}")
                continue
                
            # 2. Convertir a Escala de Grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 3. Preparar parámetros de guardado según extensión
            filename = os.path.basename(file_path)
            output_path = os.path.join(PROCESSED_DIR, filename)
            ext_lower = os.path.splitext(filename)[1].lower()
            
            encode_params = []
            
            if ext_lower in ['.jpg', '.jpeg']:
                # CALIDAD JPG: Rango 0-100.
                # 95 es el default de OpenCV (muy alto).
                # 85 es un buen balance. 70 reduce mucho el peso.
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                
            elif ext_lower == '.png':
                # COMPRESIÓN PNG: Rango 0-9.
                # 3 es default. 9 es máxima compresión (más lento, menor peso).
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]

            # 4. Guardar con parámetros
            cv2.imwrite(output_path, gray, encode_params)
            
            # 5. Comparar Tamaños
            size_original = get_size_kb(file_path)
            size_final = get_size_kb(output_path)
            diff = size_original - size_final
            
            # Imprimir feedback visual
            if size_final < size_original:
                status = f"✅ AHORRO: {diff:.1f} kB"
            else:
                status = f"⚠️ AUMENTO: {abs(diff):.1f} kB (Intenta bajar la calidad JPG a 70)"

            print(f"Procesado: {filename} | {size_original:.1f}kB -> {size_final:.1f}kB | {status}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Error procesando {file_path}: {e}")

    print("\n" + "="*40)
    print(f"🎉 Procesamiento Completado")
    print(f"   Total: {processed_count} imágenes")
    print(f"   Carpeta: {PROCESSED_DIR}")
    print("="*40)

if __name__ == "__main__":
    process_images()