import argparse
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import zipfile
import tempfile

# ==========================================
# CONFIGURACIÓN
# ==========================================
BATCH_SIZE = 32
EPOCHS_FINE_TUNE = 2
LEARNING_RATE = 1e-4

TFLITE_DIR = "models/tflite"

def parse_args():
    parser = argparse.ArgumentParser(description="Aplica poda estructurada (70%) a un modelo pre-entrenado.")
    parser.add_argument('--model_path', type=str, required=True, help="Ruta al modelo base (ej. models/checkpoints/modelo.keras)")
    parser.add_argument('--data_dir', type=str, default="data/processed/320x320", help="Ruta al directorio de datos procesados.")
    return parser.parse_args()

def load_data(data_dir):
    """Carga y normaliza los datos desde el directorio usando image_dataset_from_directory."""
    print(f"Cargando datos desde: {data_dir}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode="grayscale",
        image_size=(320, 320),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=(320, 320),
        batch_size=BATCH_SIZE
    )

    # Normalizar a [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Opcional pero recomendado para rendimiento
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def get_gzipped_model_size(file):
    """Devuelve el tamaño del modelo comprimido (gzipped) en bytes."""
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

def train_and_save(model, name, train_ds, val_ds):
    """Ajusta (fine-tunes), limpia y convierte/guarda el modelo."""
    print(f"\n--- Procesando: {name} ---")
    
    # 1. Compilar y Ajustar (Fine-tune)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    
    print("Ajustando (Fine-tuning)...")
    model.fit(
        train_ds,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # 2. Eliminar Envoltorios (Wrappers)
    print("Eliminando envoltorios de poda...")
    model_export = tfmot.sparsity.keras.strip_pruning(model)
    
    # 3. Guardar TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model_export)
    tflite_model = converter.convert()
    
    path = os.path.join(TFLITE_DIR, f"{name}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
        
    # Obtener tamaños
    size_original = len(tflite_model)
    size_zipped = get_gzipped_model_size(path)
    
    print(f"Guadado en: {path}")
    print(f"Tamaño: {size_original/1024:.2f} KB")
    print(f"Tamaño Comprimido: {size_zipped/1024:.2f} KB (Tamaño aprox. de transmisión)")
    return model_export

def main():
    args = parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Modelo no encontrado en {args.model_path}")
        return

    train_ds, val_ds = load_data(args.data_dir)
    
    # Asegurar que exista el directorio TFLite
    os.makedirs(TFLITE_DIR, exist_ok=True)

    print(f"Cargando modelo base desde: {args.model_path}")
    baseline_model = keras.models.load_model(args.model_path)
    
    # ==========================================
    # PODA ESTRUCTURADA 70%
    # ==========================================
    print("\nAplicando Poda Estructurada (70% de esparsidad)")
    
    # Para poda estructurada en TF-MOT, se utiliza comúnmente block_size.
    # Por ejemplo, block_size=(1, 4) para podar bloques de 4 pesos.
    # Si recibes un error por compatibilidad con alguna capa específica,
    # puedes remover el parámetro block_size y será evaluado como poda de magnitud simple (unstructured).
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.70,
            begin_step=0,
            end_step=1000, # Ajustar basado en el tamaño de dataset * épocas
        ),
        'block_size': (1, 4), # Argumento clave para poda estructurada por bloques
        'block_pooling_type': 'AVG'
    }
    
    def apply_pruning_to_layers(layer):
        if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    model_pruned = keras.models.clone_model(
        baseline_model,
        clone_function=apply_pruning_to_layers
    )
    
    # Nombre de salida basado en el modelo de entrada
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    out_name = f"{base_name}_pruned_structured_70"
    
    train_and_save(model_pruned, out_name, train_ds, val_ds)
    print("\n¡Proceso de poda estructurada al 70% completado con éxito!")

if __name__ == "__main__":
    main()
