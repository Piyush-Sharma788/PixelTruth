import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Create an UNTRAINED MobileNetV2 model with the v2 contract shape, for tests only
mnet = MobileNetV2(include_top=False, weights=None, input_shape=(96, 96, 3))
model = Sequential([
    Rescaling(scale=1./127.5, offset=-1., input_shape=(96, 96, 3)),
    mnet,
    GlobalAveragePooling2D(),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy")
# Never overwrite the real trained model: write to a separate file and point
# PIXELTRUTH_MODEL_PATH at it if you need a random-weights model for testing.
model.save("dummy_model.h5")
print("Created dummy_model.h5 (RANDOM WEIGHTS - not for real predictions)")
