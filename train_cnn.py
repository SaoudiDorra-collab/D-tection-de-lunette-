# train_cnn_fast.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time

print("⏱️  Début de l'entraînement rapide...")
start_time = time.time()

# -----------------------------
# 1. Préparation des données OPTIMISÉE
# -----------------------------
img_size = (150, 150)  # Réduit de 224x224 à 150x150
batch = 16  # Batch size plus petit

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,  # Réduit
    width_shift_range=0.1,  # Réduit
    height_shift_range=0.1  # Réduit
)

train_data = train_gen.flow_from_directory(
    "dataset/",
    target_size=img_size,
    batch_size=batch,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    "dataset/",
    target_size=img_size,
    batch_size=batch,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print(f"📊 Classes trouvées: {train_data.class_indices}")

# -----------------------------
# 2. CNN SIMPLE et RAPIDE
# -----------------------------
model_cnn = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_cnn.summary()

# -----------------------------
# 3. Compilation OPTIMISÉE
# -----------------------------
model_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 4. Callbacks pour arrêt précoce
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# -----------------------------
# 5. Entraînement RAPIDE (max 15 epochs)
# -----------------------------
print("\n🎯 Début de l'entraînement...")
history = model_cnn.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Maximum 15 epochs
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 6. Sauvegarde et évaluation
# -----------------------------
model_cnn.save("model_cnn_fast.h5")

# Temps d'exécution
end_time = time.time()
training_time = end_time - start_time
print(f"\n✅ Entraînement terminé en {training_time:.2f} secondes")
print(f"   Soit {training_time/60:.2f} minutes")

# -----------------------------
# 7. Visualisation rapide
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_fast.png', dpi=100)
plt.show()

# -----------------------------
# 8. Évaluation finale
# -----------------------------
final_val_acc = history.history['val_accuracy'][-1]
final_train_acc = history.history['accuracy'][-1]

print(f"\n📈 Résultats finaux:")
print(f"   Accuracy entraînement: {final_train_acc:.4f}")
print(f"   Accuracy validation: {final_val_acc:.4f}")
print(f"   Différence: {abs(final_train_acc - final_val_acc):.4f}")

if final_val_acc > 0.75:
    print("🎉 Bon modèle!")
elif final_val_acc > 0.60:
    print("👍 Modèle acceptable")
else:
    print("⚠️  Modèle à améliorer")