import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle  # To save and load history

# File paths
data_dir = r"C:\Users\shali\Documents\L&D\GitHub Projects\Machine Learning\003_LIVECell_Segmentation of Cells_U-Net\processed_data_unet"
output_dir = r"C:\Users\shali\Documents\L&D\GitHub Projects\Machine Learning\003_LIVECell_Segmentation of Cells_U-Net\output_model"
images_path = os.path.join(data_dir, "images.npy")
masks_path = os.path.join(data_dir, "masks.npy")
output_model_path = os.path.join(output_dir, "unet_trained_model.h5")
history_path = os.path.join(output_dir, "training_history.pkl")  # Path to save history

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
X = np.load(images_path)  # Images
y = np.load(masks_path)   # Masks

# Check for mismatched dimensions in masks
if len(y.shape) == 3:  # Add channel dimension if missing
    print("Adding channel dimension to masks...")
    y = np.expand_dims(y, axis=-1)

# Train-validation split
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# Custom Metrics: Intersection over Union (IoU) and Dice Coefficient
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold predictions at 0.5
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (union + K.epsilon())

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold predictions at 0.5
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
    return dice

# Simplified U-Net Model
def unet(input_shape=(512, 512, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)

    # Decoder
    u3 = layers.UpSampling2D((2, 2))(bottleneck)
    u3 = layers.Concatenate()([u3, c3])
    u3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)

    u2 = layers.UpSampling2D((2, 2))(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)

    u1 = layers.UpSampling2D((2, 2))(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)

    # Output Layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u1)

    model = models.Model(inputs, outputs)
    return model

# Instantiate and compile the U-Net model
print("Building U-Net model...")
model = unet(input_shape=(512, 512, 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', iou_metric, dice_coefficient]
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16,  # Recommended batch size for balance
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save the model
model.save(output_model_path)
print(f"Model saved to: {output_model_path}")

# Save training history
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"Training history saved to: {history_path}")

# Print Training Complete Message
print("Training complete. Final model saved.")
