from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "real_and_fake_face_detection/real_vs_fake/real-vs-fake/train"
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.2

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

train = train_datagen.flow_from_directory(DATASET_PATH,
                                          class_mode="binary",
                                          target_size=IMAGE_SIZE,
                                          batch_size=BATCH_SIZE,
                                          subset="training")

val = val_datagen.flow_from_directory(DATASET_PATH,
                                      class_mode="binary",
                                      target_size=IMAGE_SIZE,
                                      batch_size=BATCH_SIZE,
                                      subset="validation")

mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(2, activation="softmax")])

model.layers[0].trainable = False
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    else:
        return 0.0001

lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

hist = model.fit(train,
                 epochs=EPOCHS,
                 callbacks=[lr_callbacks],
                 validation_data=val)

model.save('deepfake_detection_model.h5')
print("✅ Model saved!")

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(EPOCHS)

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