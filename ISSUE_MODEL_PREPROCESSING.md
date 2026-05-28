# Bug: Training and inference use incompatible MobileNetV2 input scaling

## Summary

The project does not define one stable preprocessing contract for the saved
model. A model produced by `train.py` is trained on MobileNetV2
`preprocess_input` values, but the inference pipeline passes raw RGB float
values in the `0..255` range. The newer training scripts define a third
contract by placing `Rescaling(1./255)` before the pretrained MobileNetV2
backbone.

This can make inference results unreliable even when the application loads a
valid model successfully.

## Evidence

- `train.py` configures `ImageDataGenerator(preprocessing_function=preprocess_input)`.
- `preprocessing.py::preprocess_image_array` returns resized `float32` pixels
  without MobileNetV2 preprocessing.
- `train_v2.py` and `train_v3.py` add `Rescaling(1./255)` in the model instead
  of using MobileNetV2 `preprocess_input`.
- `README.md` identifies only one default model filename and does not state
  which preprocessing contract that artifact uses.

## Steps To Reproduce

1. Train and save a model using `train.py`.
2. Pass the same image through the training generator and through
   `predict.preprocess_image`.
3. Inspect the tensors supplied to the MobileNetV2 backbone.

## Actual Result

The training path transforms pixels using MobileNetV2 preprocessing, while
runtime inference supplies raw `0..255` values. Models produced by different
training scripts also expect different input scaling.

## Expected Result

All supported training scripts and inference entry points use one documented
input contract, and tests assert that the tensor entering the backbone matches
that contract.

## Suggested Resolution

1. Select the canonical model artifact and its required input scaling.
2. Move preprocessing into the saved model, or apply the same transformation
   consistently in every inference and training path.
3. Add an integration test comparing training-time and inference-time model
   inputs for a fixed image.
4. Document the model version and input contract in `README.md`.
