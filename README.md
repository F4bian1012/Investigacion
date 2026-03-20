# Investigación TinyML - Arduino Portenta H7

Este proyecto sigue una arquitectura **MLOps** rigurosa adaptada para Sistemas Embebidos (TinyML). El objetivo es garantizar la reproducibilidad de los experimentos, la trazabilidad de los modelos y una clara separación entre el entrenamiento (Python) y el despliegue (C++).

## 📂 Filosofía de la Estructura de Directorios

La organización de carpetas ha sido diseñada siguiendo las mejores prácticas de la industria:

### 1. `data/` - El ciclo de vida de los datos
Separamos los datos en tres estados inmutables para evitar corrupción y fugas de información:
- **`raw/`**: Datos crudos tal como se obtuvieron del sensor o fuente. **Nunca se sobreescriben.**
- **`processed/`**: Datos limpios, normalizados y listos para entrar al modelo.
- **`augmented/`**: Versiones generadas artificialmente para robustecer el entrenamiento (ruido añadido, cambios de tono, etc.).

### 2. `models/` - Gestión de Artefactos
- **`checkpoints/`**: Guardamos los modelos "pesados" de entrenamiento (formato `.keras` o `.h5`). Estos contienen el estado completo del optimizador.
- **`tflite/`**: Aquí residen únicamente los modelos cuantizados y optimizados para el microcontrolador (`.tflite`) y sus conversiones a arreglos de C (`model.h`).

### 3. `src/` vs `deployment/`
- **`src/`**: Contiene todo el código Python para la "ciencia" del proyecto (entrenamiento, evaluación, scripts de utilidad). Es el entorno del Data Scientist.
- **`deployment/arduino_project/`**: Contiene el código fuente C++ final que se cargará en la Portenta H7. Es el entorno del Ingeniero Embebido. Mantener esto separado evita conflictos de dependencias entre Python y C++.

### 4. `logs/` y `config/`
- **`logs/`**: Almacena gráficas de pérdidas (loss curves), métricas de precisión y registros de experimentos. Fundamental para comparar qué iteración del modelo funciona mejor.
- **`config/`**: Archivos de configuración o hiperparámetros.

## 🚀 Cómo empezar

1. Instala las dependencias de Python:
   ```bash
   pip install -r requirements.txt
   ```
2. Realiza tus experimentos en `notebooks/` o scripts en `src/`.
4. Copia el arreglo C generado a `deployment/arduino_project/` para compilarlo en Arduino IDE.

## 🔄 Flujo de Trabajo Principal (MLOps)

Scripts orquestadores para el ciclo de vida completo del modelo:

### 1. Entrenamiento Modelo Base (`src/train_model.py`)
Entrena el modelo base (CNN) con datos de Fashion MNIST normalizados.
- **Entrada:** Dataset Fashion MNIST (descarga automática).
- **Salida:** `models/checkpoints/best_model.keras`.
- **Uso:** `python src/train_model.py`

### 1.1 Entrenamiento MobileNet - Datos Personalizados (`src/train_mobilenet.py`)
Entrena un modelo de la familia MobileNet con Transfer Learning sobre los datos procesados, ideal para inferencia en el Portenta H7.

**Entradas Automáticas:**
- **Entrada:** Las imágenes se cargan automáticamente desde `data/processed/{width}x{height}/` según la resolución indicada.

**Salidas (Checkpoints y Logs Dinámicos):** 
Se guardan de forma dinámica incorporando los parámetros en el nombre del archivo para facilitar el tracking y versionado de experimentos:
- **Modelo:** `models/checkpoints/{base_model}+{batch_size}+{epochs}+{learning_rate}+{validation_split}+{width}+{height}.keras`
- **Gráfica:** `tensorboard_logs/{base_model}_training_history+{batch_size}+{epochs}+{learning_rate}+{validation_split}+{width}+{height}.png`

**Hiperparámetros (Soportados por CLI):**
- `--base_model` (Opciones: `MobileNet`, `MobileNetV2`, `MobileNetV3Large`, `MobileNetV3Small` | Por defecto: `MobileNetV2`)
- `--width` y `--height` (Por defecto: 96)
- `--batch_size` (Por defecto: 32)
- `--epochs` (Por defecto: 20)
- `--learning_rate` (Por defecto: 0.0001)
- `--validation_split` (Por defecto: 0.2)

**Uso:**
```bash
# Entrenamiento con valores por defecto (MobileNetV2)
python src/train_mobilenet.py

# Uso con el modelo base original (MobileNet v1)
python src/train_mobilenet.py --base_model "MobileNet"

# Personalizando los hiperparámetros
python src/train_mobilenet.py --base_model "MobileNetV3Large" --width 96 --height 96 --batch_size 16 --epochs 30 --learning_rate 0.00005 --validation_split 0.25
```

### 2. Optimización (`src/optimize_model.py`)
Pipeline completo que aplica **Pruning** (poda) y **Cuantización Post-Entrenamiento** (PTQ) para generar modelos eficientes.
- **Proceso:** Carga el modelo `.keras` -> Poda -> Fine-tuning -> Conversión a TFLite (Int8/Float).
- **Salida:** Datos generados en `models/tflite/`.
- **Uso:** `python src/optimize_model.py`

### 3. Validación (`src/validate_model.py`)
Compara la precisión del modelo original vs. el modelo cuantizado para asegurar calidad antes del despliegue.
- **Uso:** `python src/validate_model.py`

## 🛠️ Scripts de Optimización (NUEVO)

Se han añadido scripts avanzados para aplicar técnicas de compresión de modelos, esenciales para hardware limitado:

### 1. `src/pruning_techniques.py` (Poda)
Este script aplica diferentes estrategias para reducir conexiones neuronales no esenciales:
- **Poda de Decaimiento Polinómico**: Aumenta gradualmente la dispersión durante el entrenamiento.
- **Dispersión Constante**: Mantiene un nivel fijo de "ceros" en los pesos.
- **Poda por Capas**: Aplica diferentes agresividades de poda según el tipo de capa (Conv2D vs Dense).

**Uso:**
```bash
python3 src/pruning_techniques.py
```

### 2. `src/quantization_techniques.py` (Cuantización)
Este script demuestra cómo reducir la precisión numérica de los pesos y activaciones para ahorrar memoria (Flash/RAM) y acelerar la inferencia:
- **Rango Dinámico**: Pesos int8, activaciones float32.
- **Enteros Completo (Float Fallback)**: Intenta int8, usa float si es necesario.
- **Enteros Completo (Integer Only)**: Obligatorio para MCUs sencillos (Portenta, ESP32).
- **Float16**: Reduce a la mitad el tamaño, útil para GPUs.
- **QAT (Training Aware)**: Simula la cuantización durante el entrenamiento para recuperar precisión.

**Uso:**
```bash
python3 src/quantization_techniques.py
```
> **Nota:** Requiere instalar `tensorflow-model-optimization`.

## 📸 Captura y Visualización de Imágenes

Herramientas para capturar y visualizar datos desde la cámara de la Portenta H7.

### 1. Firmware Arduino (`deployment/arduino/image_capture/image_capture.ino`)
Script para la Portenta H7 que captura imágenes en escala de grises (160x120) y las envía como bytes crudos a través del puerto serial.
- **Configuración:** QQVGA (160x120), Grayscale, 30 FPS.
- **Uso:** Cargar en la placa usando Arduino IDE.

### 2. Visualizador Python (`src/visualize_serial_image.py`)
Script para recibir y renderizar en tiempo real las imágenes enviadas por el Arduino.
- **Detección automática de puerto:** Intenta encontrar el puerto serial de la Portenta si no se especifica.
- **Renderizado:** Utiliza `matplotlib` para mostrar el stream de video.

**Uso:**
```bash
python src/visualize_serial_image.py
# O especificando el puerto manualmente:
# En mac
python src/visualize_serial_image.py --port /dev/tty.usbmodem1301 
# En windows
python src/visualize_serial_image.py --port COM7
```

### 3. Procesamiento de Imágenes (`src/process_images.py`)
Convierte imágenes a escala de grises controlando la compresión para evitar que aumente el tamaño del archivo, preparándolas para el entrenamiento.

**Uso básico (rutas por defecto):**
```bash
python src/process_images.py
```

**Uso con rutas personalizadas:**
```bash
python src/process_images.py --raw_path "ruta/a/imagenes_crudas" --path_processed "ruta/a/destino"
```

### 4. Redimensionamiento de Imágenes (`src/reshape_images.py`)
Redimensiona las imágenes procesadas al tamaño objetivo para TinyML, conservando la estructura de carpetas (clases).

**Uso básico (rutas por defecto, busca recursivamente en `data/processed/grayscale`):**
```bash
python src/reshape_images.py --width 96 --height 96
```

**Uso con directorio de entrada personalizado:**
```bash
python src/reshape_images.py --input_dir "ruta/personalizada" --width 96 --height 96
```