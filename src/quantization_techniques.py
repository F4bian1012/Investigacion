import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import pathlib

# ==========================================
# CONFIGURACIÓN
# ==========================================
BATCH_SIZE = 32
CHECKPOINT_PATH = "models/checkpoints/best_model.keras"
TFLITE_DIR = "models/tflite"

def load_data():
    """Carga y normaliza los datos de Fashion MNIST."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def representative_data_gen():
    """Generador para Conjunto de Datos Representativo (Requerido para Cuantización de Enteros)."""
    (x_train, _), _ = load_data()
    # Usar 100 imágenes para calibración
    # tf.lite.TFLiteConverter espera una lista de entradas (incluso si solo hay un tensor de entrada)
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        yield [input_value]

def save_tflite_model(tflite_model, filename):
    """Guarda el modelo TFLite en disco e imprime el tamaño."""
    path = os.path.join(TFLITE_DIR, filename)
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"   Guardado: {path} ({len(tflite_model)/1024:.2f} KB)")

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Modelo no encontrado en {CHECKPOINT_PATH}")
        return

    os.makedirs(TFLITE_DIR, exist_ok=True)
    
    print("Cargando modelo base...")
    model = keras.models.load_model(CHECKPOINT_PATH)

    # ---------------------------------------------------------
    # 1. CUANTIZACIÓN DE RANGO DINÁMICO
    # ---------------------------------------------------------
    # Ideal para: Reducción inicial de tamaño (4x), mínima pérdida de precisión, aceleración CPU.
    # Qué hace: Pesos son int8, activaciones son float32 (cuantizadas dinámicamente en tiempo de ejecución).
    print("\n[Método 1] Cuantización de Rango Dinámico")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic_model = converter.convert()
    save_tflite_model(tflite_dynamic_model, "model_dynamic_range.tflite")

    # ---------------------------------------------------------
    # 2. CUANTIZACIÓN DE ENTEROS COMPLETA (RESPALDO FLOTANTE)
    # ---------------------------------------------------------
    # Ideal para: Compatibilidad con EdgeTPU/DSP pero permitiendo ops no cuantizables en CPU como float.
    # Qué hace: Pesos y activaciones son int8. Ops sin implementación int8 usan float.
    print("\n[Método 2] Cuantización de Enteros Completa (Respaldo Flotante)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    tflite_int8_fallback_model = converter.convert()
    save_tflite_model(tflite_int8_fallback_model, "model_int8_fallback.tflite")

    # ---------------------------------------------------------
    # 3. CUANTIZACIÓN DE ENTEROS COMPLETA (SÓLO ENTEROS)
    # ---------------------------------------------------------
    # Ideal para: Microcontroladores (Arduino Portenta, ESP32) sin FPU o aceleradores distintos.
    # Qué hace: Fuerza todas las ops a ser int8. Entrada/Salida también int8.
    print("\n[Método 3] Cuantización de Enteros Completa (Sólo Enteros)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Asegurar que todas las ops estén soportadas en int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Establecer tensores de entrada y salida a int8 (crucial para algunos MCUs)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_int8_only_model = converter.convert()
        save_tflite_model(tflite_int8_only_model, "model_int8_only.tflite")
    except Exception as e:
        print(f"   ⚠️ Falló la conversión (común si el modelo tiene ops no cuantizables): {e}")

    # ---------------------------------------------------------
    # 4. CUANTIZACIÓN FLOAT16
    # ---------------------------------------------------------
    # Ideal para: GPUs (móvil/servidor) que soportan aceleración float16. Reducción ~2x.
    # Qué hace: Pesos son float16.
    print("\n[Método 4] Cuantización Float16")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16_model = converter.convert()
    save_tflite_model(tflite_fp16_model, "model_float16.tflite")

    # ---------------------------------------------------------
    # 5. ENTRENAMIENTO CONSCIENTE DE CUANTIZACIÓN (QAT)
    # ---------------------------------------------------------
    # Ideal para: Recuperar pérdida de precisión por PTQ. Simula cuantización durante el entrenamiento.
    print("\n[Método 5] Entrenamiento Consciente de Cuantización (QAT)")
    
    # Recargar modelo para que esté fresco
    model_for_qat = keras.models.load_model(CHECKPOINT_PATH)
    
    # Envolver modelo
    quant_aware_model = tfmot.quantization.keras.quantize_model(model_for_qat)
    
    # Recompilar (QAT requiere entrenamiento)
    quant_aware_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("   Ajustando modelo QAT (1 época para demo)...")
    (x_train, y_train), (x_test, y_test) = load_data()
    quant_aware_model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=1
    )
    
    # Convertir modelo QAT a TFLite (debe ser int8)
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Modelos QAT aún necesitan dataset representativo para rangos de activación que no se aprendieron?
    # De hecho, docs de tfmot dicen "Use TFLiteConverter con estrategia DEFAULT optimization."
    # Pero a menudo rep dataset es buena práctica o requerido para full int8.
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_qat_model = converter.convert()
    save_tflite_model(tflite_qat_model, "model_qat_int8.tflite")

    print("\n✅ ¡Todas las técnicas de cuantización completadas!")

if __name__ == "__main__":
    main()
