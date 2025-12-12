import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Input shape: 28x28 grayscale images (Fashion MNIST)
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10

# Training hyperparameters
# Batch size reduced for potential memory constraints simulation, 
# though on a PC we could go higher.
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Paths
LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CHECKPOINT_PATH = "models/checkpoints/best_model.keras"
PLOT_PATH = "logs/training_history.png"

# Ensure directories exist
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_data():
    """
    Loads and preprocesses the Fashion MNIST dataset.
    This dataset serves as a proxy for low-res camera input.
    """
    print("📥 Loading Fashion MNIST data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    # Crucial for neural network convergence and quantization later
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to include channel dimension (28, 28, 1)
    # Required for Conv2D layers
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"   Training samples: {x_train.shape[0]}")
    print(f"   Test samples: {x_test.shape[0]}")
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """
    Creates a lightweight CNN architecture suitable for TinyML.
    
    Philosophy for Portenta H7 (Cortex-M7):
    1. Minimize parameter count (Flash usage).
    2. Minimize intermediate activation buffers (RAM usage).
    3. Use standard layers supported by TFLite Micro.
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)),
        
        # First Convolutional Block
        # 8 filters is very small, but sufficient for simple edge detection
        # Strides=2 reduces spatial dimension early, saving RAM for next layers
        layers.Conv2D(8, (3, 3), padding='same', activation='relu', strides=2),
        
        # Second Convolutional Block
        # Flattening 14x14x8 -> 1568 values
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', strides=2),
        
        # Flattening
        # 7x7x16 = 784
        layers.Flatten(),
        
        # Dense Layer
        # 32 neurons is a balance between capacity and memory.
        # Dropout helps prevent overfitting on this small capacity.
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output Layer
        # Softmax for multi-class classification
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def plot_history(history):
    """
    Plots training and validation metrics.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.savefig(PLOT_PATH)
    print(f"📊 Training plots saved to {PLOT_PATH}")

def main():
    # 1. Load Data
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2. Build Model
    model = create_model()
    model.summary()

    # 3. Callbacks
    callbacks = [
        # Save only the best model based on validation accuracy
        keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        # Stop if no improvement after 3 epochs to save time
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 4. Train
    print("🚀 Starting training...")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    # 5. Evaluate
    print("\n⚖️ Evaluating best model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")

    # 6. Visualize
    plot_history(history)
    print(f"✅ Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
