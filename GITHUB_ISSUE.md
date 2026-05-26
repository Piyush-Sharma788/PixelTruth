# Bug Report: Committed Merge Conflicts and Broken Inference/Training Logic

## 1. Summary of Issues

Three critical bugs were identified in the codebase:
1. **Committed Git Merge Conflicts in `app.py`**: The application file `app.py` was committed with raw git conflict markers (`=======`, `>>>>>>> upstream/main`) and duplicate UI sections. This causes compilation/syntax errors and prevents the Streamlit app from launching.
2. **Inverted and Broken Labeling/Prediction Logic in `app.py`**:
   - The manual decoding logic in the files loop used `label = "Real" if class_label == 0 else "Fake"`. However, the model classes are mapped alphabetically: `0` represents "Fake" and `1` represents "Real". This inverted the classification results.
   - The code used `class_label = int(np.argmax(prediction, axis=1)[0])` to extract the winning class index. For sigmoid models (such as the default model trained in `train.py`), the output shape is `(1, 1)`, which means `np.argmax(..., axis=1)` always returns index `0`. Consequently, every image was classified as "Real".
3. **Missing Import in `train.py`**: The training script `train.py` uses Keras's `ImageDataGenerator` but fails to import it, leading to a `NameError: name 'ImageDataGenerator' is not defined` when executed.

---

## 2. Steps to Reproduce

### Issue 1 & 2 (Streamlit App)
1. Run `streamlit run app.py` on the `upstream/main` or current branch.
2. **Result**: An `IndentationError` / `SyntaxError` is raised due to conflict markers.
3. If conflict markers are bypassed, uploading a deepfake face photo with a sigmoid model will output a "Real" classification with extremely low confidence because of the `argmax` and label-inversion bugs.

### Issue 3 (Training Code)
1. Run `python train.py`.
2. **Result**: A crash occurs immediately:
   ```
   NameError: name 'ImageDataGenerator' is not defined
   ```

---

## 3. Proposed Fixes & Implementation

All of these issues have been resolved locally on this branch:
1. **Resolved `app.py` Conflicts**: Consolidated the duplicate UI blocks and removed the git conflict markers.
2. **Fixed Prediction Logic in `app.py`**: Replaced the manual and buggy `np.argmax` logic in the files loop with the core `predict_image` utility from `inference.py` (which correctly handles both sigmoid and 2-class softmax models).
3. **Fixed `train.py` Imports**: Added the missing import statement:
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   ```

All unit tests compile and pass successfully (`pytest` results: 17 passed).
