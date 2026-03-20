import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
IMG_WIDTH = 28
IMG_HEIGHT = 28
BATCH_SIZE = 32
EPOCHS_FINE_TUNE = 2 # Short fine-tuning
LEARNING_RATE = 1e-4 # Lower LR for fine-tuning

# Paths
CHECKPOINT_PATH = "models/checkpoints/best_model.keras"
TFLITE_DIR = "models/tflite"

def load_data():
    """
    Loads data for representative dataset generation and fine-tuning.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def representative_data_gen():
    """
    Generator function for the Representative Dataset.
    
    CRITICAL FOR INT8 QUANTIZATION:
    To quantize activations (which are dynamic) to 8-bit integers,
    the converter needs to observe the range of values flow through the model
    with real data. This generator provides that data.
    """
    (x_train, _), _ = load_data()
    # Use a small subset of 100 images for calibration
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        yield [input_value]

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Model not found at {CHECKPOINT_PATH}. Train the model first!")
        return

    # 1. Load Baseline Model
    print("📥 Loading baseline model...")
    model = keras.models.load_model(CHECKPOINT_PATH)

    # 2. Pruning
    print("✂️ Applying Pruning...")
    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.50, # Prune 50% of connections
            begin_step=0,
            end_step=1000 # Short schedule for demo
        )
    }

    # Wrap the model for pruning
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Re-compile
    model_for_pruning.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Fine-tune
    print("↻ Fine-tuning pruned model...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Logs summarization just for ensuring it runs, typically viewed in TensorBoard
    ]
    
    model_for_pruning.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    # Strip pruning wrapper to get a clean model for export
    print("🧹 Stripping pruning wrappers...")
    model_pruned_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    # 3. Conversion & Quantization
    print("📦 Converting to TFLite...")

    # A) FLOAT32 TFLite (Baseline)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path_float = os.path.join(TFLITE_DIR, "model_float32.tflite")
    with open(path_float, "wb") as f:
        f.write(tflite_model)
    print(f"   Saved: {path_float} ({len(tflite_model)/1024:.1f} KB)")

    # B) Pruned TFLite (Float32 but sparse)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_pruned_export)
    tflite_pruned_model = converter.convert()
    path_pruned = os.path.join(TFLITE_DIR, "model_pruned.tflite")
    with open(path_pruned, "wb") as f:
        f.write(tflite_pruned_model)
    print(f"   Saved: {path_pruned} ({len(tflite_pruned_model)/1024:.1f} KB)")

    # C) INT8 Quantization (Full Integer)
    # This is the 'gold standard' for microcontrollers
    converter = tf.lite.TFLiteConverter.from_keras_model(model_pruned_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Provide representative dataset
    converter.representative_dataset = representative_data_gen
    
    # Ensure intermediate operations are also quantized to int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Ensure input/output are int8 (optional, but good for pure int8 pipeline)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_quant_model = converter.convert()
        path_quant = os.path.join(TFLITE_DIR, "model_pruned_quant_int8.tflite")
        with open(path_quant, "wb") as f:
            f.write(tflite_quant_model)
        print(f"   Saved: {path_quant} ({len(tflite_quant_model)/1024:.1f} KB)")
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")

    print("\n✅ Optimization pipeline complete!")

if __name__ == "__main__":
    main()
