from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

dataset_path = "real_and_fake_face_detection/real_vs_fake/real-vs-fake/train"

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    rescale=1./255,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = train_datagen.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=128,
                                          subset="training")

val = val_datagen.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=128,
                                          subset="validation")

print(f"Class indices: {train.class_indices}")

mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
mnet.trainable = False

model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.4),
                    Dense(128, activation="relu"),
                    Dropout(0.2),
                    Dense(2, activation="softmax")])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=["accuracy"])

# Class weights — fake aur real ko equal importance
class_weight = {0: 1.5, 1: 1.0}

# Save the model only when validation loss improves
checkpoint = ModelCheckpoint(
    filepath='deepfake_detection_model.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

# Stop training if validation loss hasn't improved for 3 consecutive epochs
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True,
    verbose=1
)

callbacks_list = [checkpoint, early_stop]

print("Phase 1 - Frozen (5 epochs)")
hist1 = model.fit(train, epochs=5, validation_data=val, class_weight=class_weight, callbacks=callbacks_list)

# Unfreeze
mnet.trainable = True
for layer in mnet.layers[:-30]:
    layer.trainable = False

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(0.00001),
              metrics=["accuracy"])

print("Phase 2 - Unfreeze (5 epochs)")
hist2 = model.fit(train, epochs=5, validation_data=val, class_weight=class_weight, callbacks=callbacks_list)


# Graphs
train_acc  = hist1.history['accuracy']  + hist2.history['accuracy']
val_acc    = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
train_loss = hist1.history['loss'] + hist2.history['loss']
val_loss   = hist1.history['val_loss'] + hist2.history['val_loss']
xc = range(len(train_loss))

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
print("Graphs saved!")
