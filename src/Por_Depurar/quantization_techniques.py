import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize trained model to TFLite format")
    parser.add_argument("--base_model", type=str, default="MobileNet", help="Base model architecture (e.g. MobileNet, MobileNetV2)")
    parser.add_argument('--width', type=int, default=96, help="Image width")
    parser.add_argument('--height', type=int, default=96, help="Image height")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=20, help="Epochs config used in training (to locate model)")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate used in training")
    parser.add_argument('--validation_split', type=float, default=0.2, help="Validation split used")
    parser.add_argument('--model_path', type=str, default=None, help="Ruta al modelo entrenado")
    parser.add_argument('--data_dir', type=str, default=None, help="Ruta al directorio de datos")
    parser.add_argument('--tflite_dir', type=str, default="models/tflite", help="Directorio de salida para modelos TFLite")
    
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = f"models/checkpoints/{args.base_model}+{args.batch_size}+{args.epochs}+{args.learning_rate}+{args.validation_split}+{args.width}+{args.height}.keras"
        
    if args.data_dir is None:
        args.data_dir = f"data/processed/{args.width}x{args.height}"
        
    return args

def load_data(data_dir, width, height, batch_size):
    """Carga los datos del proyecto para cuantización."""
    print(f"Cargando dataset desde {data_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode='grayscale',
        image_size=(height, width),
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode='grayscale',
        image_size=(height, width),
        batch_size=batch_size
    )
    return train_ds, val_ds

def make_representative_data_gen(dataset):
    """Generador para Conjunto de Datos Representativo (Requerido para Cuantización de Enteros)."""
    def representative_data_gen():
        # Tomar 100 imágenes para calibración
        # tf.lite.TFLiteConverter espera una lista de entradas
        for images, _ in dataset.unbatch().batch(1).take(100):
            yield [images]
    return representative_data_gen

def save_tflite_model(tflite_model, filename, tflite_dir):
    """Guarda el modelo TFLite en disco e imprime el tamaño."""
    path = os.path.join(tflite_dir, filename)
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"   Guardado: {path} ({(len(tflite_model)/1024):.2f} KB)")

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Modelo no encontrado en {args.model_path}")
        return

    if not os.path.exists(args.data_dir):
        print(f"Error: Directorio de datos no encontrado en {args.data_dir}")
        return

    os.makedirs(args.tflite_dir, exist_ok=True)
    
    print(f"Cargando modelo base desde {args.model_path}...")
    model = keras.models.load_model(args.model_path)

    # Cargar datos e inicializar generador representativo
    train_ds, val_ds = load_data(args.data_dir, args.width, args.height, args.batch_size)
    rep_gen = make_representative_data_gen(train_ds)

    # Modificar el basename para nombrar los modelos tflite según la variante
    base = os.path.splitext(os.path.basename(args.model_path))[0]

    # ---------------------------------------------------------
    # 1. CUANTIZACIÓN DE RANGO DINÁMICO
    # ---------------------------------------------------------
    print("\n[Método 1] Cuantización de Rango Dinámico")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic_model = converter.convert()
    save_tflite_model(tflite_dynamic_model, f"{base}_dynamic_range.tflite", args.tflite_dir)

    # ---------------------------------------------------------
    # 2. CUANTIZACIÓN DE ENTEROS COMPLETA (RESPALDO FLOTANTE)
    # ---------------------------------------------------------
    print("\n[Método 2] Cuantización de Enteros Completa (Respaldo Flotante)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    tflite_int8_fallback_model = converter.convert()
    save_tflite_model(tflite_int8_fallback_model, f"{base}_int8_fallback.tflite", args.tflite_dir)

    # ---------------------------------------------------------
    # 3. CUANTIZACIÓN DE ENTEROS COMPLETA (SÓLO ENTEROS)
    # ---------------------------------------------------------
    print("\n[Método 3] Cuantización de Enteros Completa (Sólo Enteros)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_int8_only_model = converter.convert()
        save_tflite_model(tflite_int8_only_model, f"{base}_int8_only.tflite", args.tflite_dir)
    except Exception as e:
        print(f"   ⚠️ Falló la conversión: {e}")

    # ---------------------------------------------------------
    # 4. CUANTIZACIÓN FLOAT16
    # ---------------------------------------------------------
    print("\n[Método 4] Cuantización Float16")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16_model = converter.convert()
    save_tflite_model(tflite_fp16_model, f"{base}_float16.tflite", args.tflite_dir)

    # ---------------------------------------------------------
    # 5. ENTRENAMIENTO CONSCIENTE DE CUANTIZACIÓN (QAT)
    # ---------------------------------------------------------
    print("\n[Método 5] Entrenamiento Consciente de Cuantización (QAT)")
    
    model_for_qat = keras.models.load_model(args.model_path)
    
    try:
        quant_aware_model = tfmot.quantization.keras.quantize_model(model_for_qat)
        
        num_classes = len(train_ds.class_names)
        loss_fn = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"

        quant_aware_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=loss_fn,
            metrics=["accuracy"]
        )
        
        print("   Ajustando modelo QAT (1 época para demo)...")
        quant_aware_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            verbose=1
        )
        
        converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_qat_model = converter.convert()
        save_tflite_model(tflite_qat_model, f"{base}_qat_int8.tflite", args.tflite_dir)
        
    except Exception as e:
        print(f"   ⚠️ Falló el Entrenamiento Consciente de Cuantización (QAT): {e}")

    print("\n✅ ¡Todas las técnicas de cuantización procesadas!")

if __name__ == "__main__":
    main()
