"""Canonical PixelTruth training script (model contract v2.0.0).

Differences from train.py / train_v2.py / train_v3.py (kept for history):

* Every training and validation image goes through
  ``preprocessing.prepare_model_input`` -- the exact function the Streamlit
  app, FastAPI and CLI use at inference time (Haar face crop with full-image
  fallback, RGB, 96x96, float32 in [0, 255]).  This removes the train /
  inference mismatch where only inference cropped faces.
* Data comes from ``scripts/prepare_dataset.py`` (``data/faces/...``).
* The calibration temperature is fitted on the validation split and written
  to ``deepfake_detection_model.json`` together with the model's SHA-256.
* Training curves are written to Figure_1.png / Figure_2.png.

Usage::

    python scripts/prepare_dataset.py
    python experiments/train_v4.py --data data/faces
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import IMAGE_SIZE, MODEL_VERSION, SUPPORTED_EXTENSIONS  # noqa: E402

CLASS_NAMES = ("fake", "real")  # index 0 = fake, 1 = real (alphabetical, as Keras)


# ---------------------------------------------------------------------------
# Data loading through the canonical inference pipeline
# ---------------------------------------------------------------------------

def _load_one(path: str):
    from preprocessing import decode_image_bytes, prepare_model_input

    with open(path, "rb") as fh:
        bgr = decode_image_bytes(fh.read())
    batch, _, face_box = prepare_model_input(bgr)
    # batch is a float32 copy of a uint8 resize, so the uint8 cast is lossless.
    return batch[0].astype(np.uint8), face_box is not None


def load_split(split_dir: Path, cache_dir: Path, limit_per_class: int | None, workers: int):
    tag = f"{split_dir.name}_{limit_per_class or 'all'}"
    x_cache, y_cache = cache_dir / f"{tag}_x.npy", cache_dir / f"{tag}_y.npy"
    if x_cache.exists() and y_cache.exists():
        print(f"[data] {split_dir.name}: using cache {x_cache}")
        return np.load(x_cache), np.load(y_cache)

    paths, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        files = sorted(
            str(p) for p in (split_dir / name).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if limit_per_class:
            files = files[:limit_per_class]
        paths += files
        labels += [label] * len(files)

    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_load_one, paths, chunksize=64)
    x = np.stack([r[0] for r in results])
    y = np.asarray(labels, dtype=np.float32)
    faces = sum(r[1] for r in results)
    print(
        f"[data] {split_dir.name}: {len(paths)} images in {time.time() - t0:.0f}s, "
        f"face detected in {faces / len(paths):.1%}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_cache, x)
    np.save(y_cache, y)
    return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model():
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.layers import (
        BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input,
        RandomFlip, Rescaling,
    )

    h, w = IMAGE_SIZE[1], IMAGE_SIZE[0]
    backbone = MobileNetV2(include_top=False, weights="imagenet", input_shape=(h, w, 3))
    backbone.trainable = False
    model = Sequential([
        Input(shape=(h, w, 3)),
        RandomFlip("horizontal"),
        # Contract: raw [0, 255] in, MobileNetV2 expects [-1, 1].
        Rescaling(scale=1.0 / 127.5, offset=-1.0),
        backbone,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ], name="pixeltruth_mobilenetv2")
    return model, backbone, tf


def fit_temperature(p_real: np.ndarray, y: np.ndarray) -> float:
    """Grid-search the temperature that minimises validation NLL."""
    eps = 1e-7
    p = np.clip(p_real, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    best_t, best_nll = 1.0, np.inf
    for t in np.round(np.arange(0.5, 5.01, 0.05), 2):
        q = np.clip(1 / (1 + np.exp(-logits / t)), eps, 1 - eps)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def save_plots(history: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    for fname, key, title in (
        ("Figure_1.png", "loss", "Train vs Validation Loss"),
        ("Figure_2.png", "accuracy", "Train vs Validation Accuracy"),
    ):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history[key], marker="o")
        plt.plot(epochs, history[f"val_{key}"], marker="o")
        if history.get("phase_boundary"):
            plt.axvline(history["phase_boundary"] + 0.5, color="grey", ls="--", lw=1)
        plt.xlabel("Epoch")
        plt.ylabel(key.capitalize())
        plt.title(f"{title} (model {MODEL_VERSION})")
        plt.grid(True)
        plt.legend(["Train", "Validation"])
        plt.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the PixelTruth model (contract v2).")
    ap.add_argument("--data", default="data/faces")
    ap.add_argument("--out", default="deepfake_detection_model.h5")
    ap.add_argument("--train-limit", type=int, default=None, help="max images per class for training")
    ap.add_argument("--val-limit", type=int, default=None, help="max images per class for validation")
    ap.add_argument("--frozen-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=8)
    ap.add_argument("--unfreeze", type=int, default=60, help="backbone layers to unfreeze in phase 2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    data = Path(args.data)
    cache = data / ".cache"
    x_tr, y_tr = load_split(data / "train", cache, args.train_limit, args.workers)
    x_va, y_va = load_split(data / "validation", cache, args.val_limit, args.workers)
    print(f"[data] train={len(y_tr)} (real={int(y_tr.sum())}), val={len(y_va)} (real={int(y_va.sum())})")

    model, backbone, tf = build_model()
    tf.keras.utils.set_random_seed(args.seed)

    def ds(x, y, shuffle):
        d = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            d = d.shuffle(len(y), seed=args.seed, reshuffle_each_iteration=True)
        return d.batch(args.batch_size).map(
            lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = ds(x_tr, y_tr, True), ds(x_va, y_va, False)
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    best_path = Path(args.out).with_suffix(".best.weights.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(best_path), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1
    )

    # Phase 1: frozen backbone, train the head.
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=metrics)
    model.summary()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.frozen_epochs, callbacks=[ckpt])

    # Phase 2: unfreeze the top of the backbone. BatchNorm layers stay frozen
    # (inference mode) so their ImageNet statistics are not destroyed.
    backbone.trainable = True
    for layer in backbone.layers[:-args.unfreeze]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=metrics)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, callbacks=[ckpt, early, lr])

    model.load_weights(str(best_path))
    best_path.unlink(missing_ok=True)

    # Calibrate on validation, then report validation metrics.
    p_val = model.predict(val_ds, verbose=0).reshape(-1)
    temperature = fit_temperature(p_val, y_va)
    val_acc = float(np.mean((p_val >= 0.5) == (y_va == 1)))
    print(f"[calib] fitted temperature T={temperature}  val_acc={val_acc:.4f}")

    out = Path(args.out)
    model.save(out, include_optimizer=False)
    digest = sha256(out)

    history = {k: h1.history[k] + h2.history[k] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    history["phase_boundary"] = len(h1.history["loss"])
    save_plots(history, ROOT)

    card = {
        "model_version": MODEL_VERSION,
        "file": out.name,
        "sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "MobileNetV2 (ImageNet) + Dense head, sigmoid output = P(Real)",
        "input_contract": "preprocessing.prepare_model_input: Haar face crop (fallback full image) -> RGB 96x96 float32 [0,255]",
        "class_index": {"Fake": 0, "Real": 1},
        "dataset": "140k Real and Fake Faces (FFHQ real vs StyleGAN fake), HF mirror TheKernel01/140k-Real-and-Fake-Faces",
        "train_images": int(len(y_tr)),
        "validation_images": int(len(y_va)),
        "validation_accuracy": round(val_acc, 4),
        "calibration_temperature": temperature,
        "epochs_run": len(history["loss"]),
        "args": vars(args),
        "history": history,
    }
    out.with_suffix(".json").write_text(json.dumps(card, indent=2))
    print(f"[done] saved {out} (sha256 {digest}) and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
