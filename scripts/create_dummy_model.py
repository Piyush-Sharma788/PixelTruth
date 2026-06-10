import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Create a lightweight MobileNetV2 model structure matching train_v3.py
mnet = MobileNetV2(include_top=False, weights=None, input_shape=(96, 96, 3))
model = Sequential([
    Rescaling(scale=1./127.5, offset=-1., input_shape=(96, 96, 3)),
    mnet,
    GlobalAveragePooling2D(),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy")
model.save("deepfake_detection_model.h5")
print("Created dummy model successfully!")
