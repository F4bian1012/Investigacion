import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import argparse

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# MobileNetV2 expects at least 32x32, but 96x96 is a good balance for TinyML
IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3) 

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001 # Lower learning rate for fine-tuning
VALIDATION_SPLIT = 0.2

# Paths
DATA_DIR = "data/raw"
LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CHECKPOINT_PATH = "models/checkpoints/mobilenet_v2.keras"
PLOT_PATH = "logs/mobilenet_training_history.png"

# Ensure directories exist
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_custom_data():
    """
    Loads images from data/raw using tf.keras.preprocessing.
    Expected structure:
    data/raw/
        class_a/
            img1.jpg
        class_b/
            img2.jpg
    """
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"❌ Error: {DATA_DIR} is empty or does not exist.")
        print("Please organize your images in subfolders by class inside data/raw/")
        return None, None, None

    print(f"📥 Loading data from {DATA_DIR}...")
    
    # Load Training Data
    # logic to handle if images are directly in DATA_DIR or in subfolders
    # But image_dataset_from_directory expects subfolders for classes usually.
    # If user puts images directly in data/raw, we might interpret it as one class.
    # However, Keras requires 'labels="inferred"' to use subdirectories.
    # We will stick to the requirement: data/raw/placa/*.jpg
    
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )
    except ValueError as e:
        print(f"❌ Data Loading Error: {e}")
        print("💡 Ensure you have a subfolder for your class, e.g., 'data/raw/placa/image.jpg'.")
        return None, None, None

    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")
    
    if len(class_names) == 1:
        print("\n⚠️ WARNING: Only 1 class found ('{}').".format(class_names[0]))
        print("   Training a classifier with ONLY positive samples will result in a model")
        print("   that predicts this class for EVERYTHING (Accuracy will be trivial).")
        print("   You should add a 'background' or 'negative' class folder with random images")
        print("   to teach the model what is NOT a '{}'.\n".format(class_names[0]))

    # Autotune for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def create_mobilenet_model(num_classes):
    """
    Creates a MobileNetV2 model using Transfer Learning.
    """
    print(f"🏗️ Building MobileNetV2 model for {num_classes} classes...")

    # Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet',
        alpha=0.35 
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Handle Single-Class Case
    if num_classes == 1:
        # Binary output (Sigmoid) is standard for 2 classes, 
        # but with 1 class it effectively just learns bias.
        # However, Dense(1) is better than Dense(1, softmax) which is always 1.
        print("ℹ️ Using Binary classification configuration (Sigmoid) for single class.")
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss_fn = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss_fn = 'sparse_categorical_crossentropy'
    
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(PLOT_PATH)
    print(f"📊 Training plots saved to {PLOT_PATH}")

def main():
    # 1. Load Data
    train_ds, val_ds, class_names = load_custom_data()
    
    if train_ds is None:
        return

    num_classes = len(class_names)

    # 2. Build Model
    model = create_mobilenet_model(num_classes)
    model.summary()

    # 3. Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]

    # 4. Train
    print("🚀 Starting training...")
    try:
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks
        )

        # 5. Visualize
        plot_history(history)
        print(f"✅ Model saved to {CHECKPOINT_PATH}")
        
        # Optional: Save class names for inference
        with open("models/class_names.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        print("📝 Class names saved to models/class_names.txt")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Tip: Ensure you have enough images per class (at least 10-20 to start).")

if __name__ == "__main__":
    main()
