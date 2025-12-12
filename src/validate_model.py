import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import time

# Options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Paths
CHECKPOINT_PATH = "models/checkpoints/best_model.keras"
TFLITE_FLOAT_PATH = "models/tflite/model_float32.tflite"
TFLITE_INT8_PATH = "models/tflite/model_pruned_quant_int8.tflite"
CONFISTION_MATRIX_PATH = "logs/confusion_matrix.png"

def load_test_data():
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size / 1024 # KB

def evaluate_keras_model(model, x_test, y_test):
    print("   Running Keras Inference...")
    start_time = time.time()
    y_pred_probs = model.predict(x_test, verbose=0)
    end_time = time.time()
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    return acc, (end_time - start_time)

def evaluate_tflite_model(tflite_path, x_test, y_test):
    print(f"   Running TFLite Inference ({os.path.basename(tflite_path)})...")
    
    # Initialize Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Check if input needs quantization (int8)
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    predictions = []
    
    # Run inference one by one (simulating microcontroller loop)
    # Optimized for batch processing in Python for speed, but logic handles quantization
    start_time = time.time()
    
    for i in range(len(x_test)):
        input_data = x_test[i:i+1]
        
        if input_details['dtype'] == np.int8:
            # Quantize input: (real_value / scale) + zero_point
            input_data = (input_data / input_scale) + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
            
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        
        if output_details['dtype'] == np.int8:
            # Dequantize output: (quantized_value - zero_point) * scale
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
        predictions.append(np.argmax(output_data))
        
    end_time = time.time()
    
    acc = accuracy_score(y_test, predictions)
    return acc, (end_time - start_time), predictions

def plot_confusion_matrix(y_true, y_pred, classes):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, ftm='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Int8 Quantized Model)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(CONFISTION_MATRIX_PATH)
    print(f"📊 Confusion matrix saved to {CONFISTION_MATRIX_PATH}")

def main():
    if not os.path.exists(TFLITE_INT8_PATH):
        print("❌ Models not found. Run optimize_model.py first.")
        return

    print("🚀 Starting Model Validation Benchmarking...")
    x_test, y_test = load_test_data()
    # Use only 1000 samples for TFLite speed if needed, but here we do full set (10k)
    # x_test, y_test = x_test[:1000], y_test[:1000]

    results = []

    # 1. Keras Model
    keras_model = keras.models.load_model(CHECKPOINT_PATH)
    acc_keras, time_keras = evaluate_keras_model(keras_model, x_test, y_test)
    size_keras = get_file_size(CHECKPOINT_PATH)
    results.append(["Keras (Original)", f"{acc_keras*100:.2f}%", f"{size_keras:.1f} KB", "1.0x"])

    # 2. TFLite Float32
    acc_float, time_float, _ = evaluate_tflite_model(TFLITE_FLOAT_PATH, x_test, y_test)
    size_float = get_file_size(TFLITE_FLOAT_PATH)
    results.append(["TFLite (Float32)", f"{acc_float*100:.2f}%", f"{size_float:.1f} KB", f"{size_keras/size_float:.1f}x"])

    # 3. TFLite Int8
    acc_int8, time_int8, preds_int8 = evaluate_tflite_model(TFLITE_INT8_PATH, x_test, y_test)
    size_int8 = get_file_size(TFLITE_INT8_PATH)
    results.append(["TFLite (Int8)", f"{acc_int8*100:.2f}%", f"{size_int8:.1f} KB", f"{size_keras/size_int8:.1f}x"])

    # Report
    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Size (Disk)", "Reduction Factor"])
    print("\n🏆 BENCHMARK RESULTS 🏆")
    print(df_results.to_markdown(index=False))

    # Confusion Matrix for Final Model
    fashion_mnist_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    plot_confusion_matrix(y_test, preds_int8, fashion_mnist_labels)

    print("\n✅ Validation complete!")

    if acc_int8 < 0.80:
        print("⚠️ WARNING: Int8 accuracy is below 80%. Consider less aggressive pruning.")
    else:
        print("🎉 SUCCESS: Int8 model is accurate and ready for deployment.")

if __name__ == "__main__":
    main()
