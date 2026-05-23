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

model = Sequential([
    RandomFlip("horizontal"),
    Rescaling(1./255),
    mnet,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
    ])

mnet.trainable = False
model.compile(loss="binary_crossentropy",
               optimizer="adam",
               metrics=["accuracy"])
model.summary()

def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    else:
        return 0.0001

lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

hist = model.fit(
    train_ds,
    epochs=10,
    callbacks=[lr_callbacks],
    validation_data=val_ds
)

model.save('deepfake_detection_model.h5')
print("✅ Model saved!")

epochs = 10
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.savefig('Figure_1.png')

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.savefig('Figure_2.png')
print("Graphs saved!")
