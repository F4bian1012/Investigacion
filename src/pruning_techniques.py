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

# Rutas
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

def get_gzipped_model_size(file):
    """Devuelve el tamaño del modelo comprimido (gzipped) en bytes."""
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

def train_and_save(model, name, x_train, y_train, x_test, y_test):
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
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=(x_test, y_test),
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
    
    print(f"guardado en {path}")
    print(f"Tamaño: {size_original/1024:.2f} KB")
    print(f"Tamaño Comprimido: {size_zipped/1024:.2f} KB (Tamaño aprox. de transmisión)")
    return model_export

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Modelo no encontrado en {CHECKPOINT_PATH}")
        return

    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Asegurar que exista el directorio TFLite
    os.makedirs(TFLITE_DIR, exist_ok=True)

    print("Cargando modelo base...")
    baseline_model = keras.models.load_model(CHECKPOINT_PATH)
    
    # ==========================================
    # 1. DECAIMIENTO POLINÓMICO (Estándar)
    # ==========================================
    print("\n[Método 1] Poda de Decaimiento Polinómico")
    # Comienza en 0% de dispersión, termina en 50% de dispersión sobre los pasos de entrenamiento
    pruning_params_poly = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.50,
            begin_step=0,
            end_step=1000  # Ajustar basado en tamaño del dataset/épocas
        )
    }
    model_poly = tfmot.sparsity.keras.prune_low_magnitude(baseline_model, **pruning_params_poly)
    train_and_save(model_poly, "pruned_polynomial", x_train, y_train, x_test, y_test)

    # ==========================================
    # 2. DISPERSIÓN CONSTANTE
    # ==========================================
    print("\n[Método 2] Poda de Dispersión Constante")
    # Mantiene 50% de dispersión durante todo el entrenamiento
    baseline_model = keras.models.load_model(CHECKPOINT_PATH) # Recargar fresco
    pruning_params_const = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=0.5,
            begin_step=0,
            end_step=-1, # Poda durante todo el entrenamiento
            frequency=100
        )
    }
    model_const = tfmot.sparsity.keras.prune_low_magnitude(baseline_model, **pruning_params_const)
    train_and_save(model_const, "pruned_constant", x_train, y_train, x_test, y_test)

    # ==========================================
    # 3. PODA ESPECÍFICA POR CAPA
    # ==========================================
    print("\n[Método 3] Poda Específica por Capa")
    # Config personalizada: menos poda en capas tempranas (extracción de características), más en capas de densidad
    baseline_model = keras.models.load_model(CHECKPOINT_PATH) # Recargar fresco
    
    def apply_pruning_to_layers(layer):
        if isinstance(layer, keras.layers.Dense):
            # Podar capas densas fuertemente (ej. 70%)
            return tfmot.sparsity.keras.prune_low_magnitude(
                layer, 
                **{'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0, final_sparsity=0.70, begin_step=0, end_step=1000)}
            )
        elif isinstance(layer, keras.layers.Conv2D):
            # Podar capas Conv ligeramente (ej. 30%)
            return tfmot.sparsity.keras.prune_low_magnitude(
                layer,
                **{'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0, final_sparsity=0.30, begin_step=0, end_step=1000)}
            )
        return layer

    model_custom = keras.models.clone_model(
        baseline_model,
        clone_function=apply_pruning_to_layers,
    )
    
    train_and_save(model_custom, "pruned_custom_layers", x_train, y_train, x_test, y_test)

    print("\n¡Todas las técnicas de poda completadas!")

if __name__ == "__main__":
    main()
