from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D, Rescaling, RandomFlip
import tensorflow as tf
import matplotlib.pyplot as plt


dataset_path = "real_and_fake_face_detection/real_vs_fake/real-vs-fake/train"

# Load dataset using TensorFlow data pipeline
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(96, 96),
    batch_size=128,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(96, 96),
    batch_size=128,
    label_mode="binary"
)

AUTOTUNE = tf.data.AUTOTUNE

# Improve pipeline performance with shuffle, cache and prefetch
train_ds = train_ds.shuffle(buffer_size=1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))

# Phase 1 - Frozen training
mnet.trainable = False

model = Sequential([
    RandomFlip("horizontal"),
    Rescaling(scale=1./127.5, offset=-1., input_shape=(96, 96, 3)),
    mnet,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
    ])

model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"]
)
model.summary()

print("\n🔒 Phase 1 — Frozen MobileNet (5 epochs)")
hist1 = model.fit(train_ds, epochs=5, validation_data=val_ds)

# Phase 2 - Unfreeze last 30 layers
print("\n🔓 Phase 2 — Unfreezing last 30 layers (5 epochs)")
mnet.trainable = True
for layer in mnet.layers[:-30]:
    layer.trainable = False

model.compile(
    loss="binary_crossentropy", 
    optimizer=tf.keras.optimizers.Adam(0.00001), 
    metrics=["accuracy"]
)

hist2 = model.fit(train_ds, epochs=5, validation_data=val_ds)

# Save model
model.save('deepfake_detection_model.h5')
print("✅ Model saved!")

# Combine history
train_acc  = hist1.history['accuracy']  + hist2.history['accuracy']
val_acc    = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
train_loss = hist1.history['loss'] + hist2.history['loss']
val_loss   = hist1.history['val_loss'] + hist2.history['val_loss']
epochs = 10
xc = range(epochs)

plt.figure(1, figsize=(7,5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.legend(['Train','Validation'])
plt.grid(True)
plt.savefig('Figure_1.png')

plt.figure(2, figsize=(7,5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.legend(['Train','Validation'], loc=4)
plt.grid(True)
plt.savefig('Figure_2.png')
print("✅ Graphs saved!")

