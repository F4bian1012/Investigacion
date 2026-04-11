# Arduino Portenta H7 - TensorFlow Lite Micro Inference

Este repositorio contiene el código fuente (`arduino_project.ino`) para ejecutar inferencia de modelos de Machine Learning (específicamente arquitecturas tipo MobileNet V2) utilizando **TensorFlow Lite for Microcontrollers (TFLM)** en una placa **Arduino Portenta H7**.

## 📌 Visión General

El proyecto despliega un modelo pre-entrenado (y opcionalmente cuantizado) en el microcontrolador. Dada la limitación de la memoria RAM interna (SRAM) para modelos complejos como MobileNet, este código hace un uso crítico de la **memoria SDRAM externa** nativa del Portenta H7 para alojar el *Tensor Arena*.

## 🛠️ Requisitos de Hardware y Software

### Hardware
*   **Arduino Portenta H7** (o placa compatible con arquitectura M7 y memoria SDRAM accesible).

### Software y Librerías (Arduino IDE)
*   **Chirale_TensorFlowLite**: Librería no oficial optimizada de TensorFlow Lite para microcontroladores (o librerías TFLM equivalentes que soporten MbedOS).
*   **SDRAM**: Librería nativa de núcleo Arduino mbed-os para interactuar con la memoria externa del Portenta.
*   Tener el modelo convertido de `.tflite` a formato de arreglo de C (C-byte array), y guardarlo en el archivo `model1.h` en el mismo directorio.

## 🧠 Explicación Técnica Detallada

El sketch `arduino_project.ino` contiene comentarios técnicos y está estructurado en torno al paradigma clásico embebido: la configuración inicial (`setup`) y el ciclo de inferencia continua (`loop`).

### 1. Gestión Dinámica de Memoria en SDRAM
Modelos de red neuronal profunda, incluso los de arquitectura móvil, requieren un *Tensor Arena* (el espacio de memoria contiguo donde TFLM aloja tensores de entrada, salida y búferes intermedios durante la inferencia) de gran tamaño. 
En el código se han reservado **2 MB** explícitamente usando la memoria SDRAM:
```cpp
constexpr int kTensorArenaSize = 2048 * 1024; // 2 MB (Ajustable a necesidad)
uint8_t* tensor_arena = nullptr;
...
SDRAM.begin();
tensor_arena = (uint8_t*)ea_malloc(kTensorArenaSize);
```
El uso de la función embebida `ea_malloc()` ubica este bloque en memoria externa SDRAM del Portenta H7, ya que los ~1MB de RAM interna son insuficientes para ejecutar un MobileNet completo. De fallar la asignación (retorno de puntero *nullptr*), el microcontrolador detendrá su ejecución y notificará por consola.

### 2. MicroMutableOpResolver: Optimizando el Tamaño del Binario
En vez de utilizar el agrupador general `AllOpsResolver` (el cual añade el código de *todas* las operaciones matemáticas en el ecosistema TFLM al tamaño de la memoria flash compilada), este repositorio instancia un `MicroMutableOpResolver<15>`. 
Al hacer esto, **sólo se asocian al modelo las operaciones estrictamente necesarias** requeridas por una topología clásica de MobileNet V2:
* `Conv2D`, `DepthwiseConv2D`, `Add`, `Relu`, `Relu6`, `Mean` (para GlobalAveragePooling), `Reshape`, `Pad`, `FullyConnected`, `Softmax`, y `Concatenation`.

Esta es una **buena práctica vital** en la programación para microcontroladores, porque minimiza dramáticamente cuánta memoria Flash consume tu aplicación (el footprint).

### 3. Ciclo de Ejecución e Inferencia (`loop()`)
El bucle principal implementa una prueba de banco (benchmark) para el modelo compuesto de 3 partes:

1. **Simulación de Alimentación de Datos**: Inspecciona el tipado de los Tensors (`kTfLiteInt8` si el modelo fue convertido bajo Cuantización Entera, o `kTfLiteFloat32` si no). Posteriormente rellena los píxeles (entrada) con ceros. En caso de desplegar una aplicación práctica, en este punto se interconectaría el búfer visual o de cámara (`camera.read()`).
2. **Inferencia e Instrumentación**: Arranca una lectura temporal (`millis()`) y lanza la ejecución real invocando al modelo (`interpreter->Invoke()`). Seguido de esto, detiene el cronómetro.
3. **Análisis de Resultados**: Traspone hacia el Monitor Serie el tiempo que le tomó resolver la red neuronal completa enviando una latencia (en `ms`). Asimismo accede al vector de características resultantes para proyectar el resultado de la clasificación.

## 🚀 Instrucciones de Uso y Flujo

1. Garantiza que tu modelo customizado (exportado usando `xxd` u otras herramientas desde `.tflite` a arreglo C) se llame `g_model` y esté almacenado dentro de este proyecto bajo el nombre `model1.h`.
2. Conecta tu **Arduino Portenta H7** vía interfaz USB Tipo-C e ingresa al **Arduino IDE**.
3. Selecciona el entorno de hardware correcto: _Tools_ -> _Board_ -> _Arduino Mbed OS Portenta Boards_ -> _Arduino Portenta H7 (M7 core)_.
4. Presiona el botón **Subir**.
5. Tan pronto el proceso termine, abre el **Monitor Serie** y configúralo a **115200 baudios** de transmisión.
6. Espera la confirmación _"Iniciando TensorFlow Lite Micro..."_ y _"AllocateTensors() success!"_, seguido de la latencia registrada por cada iteración.
