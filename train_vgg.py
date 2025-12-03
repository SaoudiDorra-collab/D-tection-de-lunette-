# train_vgg_fast.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

print("⏱️  Début de l'entraînement VGG16 rapide...")
start_time = time.time()

# -----------------------------
# 1. Préparation des données
# -----------------------------
img_size = (150, 150)  # Taille réduite
batch = 8  # Batch plus petit pour VGG

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    "dataset/",
    target_size=img_size,
    batch_size=batch,
    class_mode="binary",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    "dataset/",
    target_size=img_size,
    batch_size=batch,
    class_mode="binary",
    subset="validation"
)

# -----------------------------
# 2. VGG16 avec fine-tuning limité
# -----------------------------
# Charger VGG16 pré-entraîné
base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(150, 150, 3)  # Taille réduite
)

# Geler TOUTES les couches de VGG
base.trainable = False

model_vgg = Sequential([
    base,
    Flatten(),
    Dense(128, activation='relu'),  # Réduit de 256 à 128
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_vgg.summary()

# -----------------------------
# 3. Compilation
# -----------------------------
model_vgg.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 4. Entraînement TRÈS court
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
]

print("\n🎯 Entraînement des nouvelles couches...")
history = model_vgg.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # Maximum 10 epochs
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 5. Sauvegarde
# -----------------------------
model_vgg.save("model_vgg_fast.h5")

end_time = time.time()
print(f"\n✅ VGG16 entraîné en {(end_time - start_time)/60:.2f} minutes")

# -----------------------------
# 6. Visualisation
# -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy VGG16')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss VGG16')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vgg_training_fast.png', dpi=100)
plt.show()