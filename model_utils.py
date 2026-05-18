import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


DEFAULT_MODEL_PATH = "deepfake_detection_model.h5"
MODEL_PATH_ENV = "PIXELTRUTH_MODEL_PATH"
MODEL_URL_ENV = "PIXELTRUTH_MODEL_URL"
MODEL_SHA256_ENV = "PIXELTRUTH_MODEL_SHA256"


def get_model_path() -> str:
    return os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)


def get_model_url() -> str:
    return os.getenv(MODEL_URL_ENV, "").strip()


def get_model_sha256() -> str:
    return os.getenv(MODEL_SHA256_ENV, "").strip()


def sha256_of_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(file_path: str, expected_sha256: str) -> None:
    if not expected_sha256:
        return

    actual_sha256 = sha256_of_file(file_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checksum mismatch for '{file_path}'. Expected {expected_sha256}, got {actual_sha256}."
        )


def download_model_file(url: str, destination_path: str) -> str:
    if not url:
        raise ValueError("A model download URL was not provided.")

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    with urllib.request.urlopen(url) as response, open(temp_path, "wb") as output_file:
        shutil.copyfileobj(response, output_file)

    temp_path.replace(destination)
    return str(destination)


def ensure_model_file(
    model_path: str | None = None,
    model_url: str | None = None,
    model_sha256: str | None = None,
    download_if_missing: bool = True,
) -> str:
    resolved_model_path = model_path or get_model_path()
    resolved_model_url = get_model_url() if model_url is None else model_url.strip()
    resolved_model_sha256 = get_model_sha256() if model_sha256 is None else model_sha256.strip()

    if os.path.exists(resolved_model_path):
        if resolved_model_sha256:
            verify_sha256(resolved_model_path, resolved_model_sha256)
        return resolved_model_path

    if download_if_missing and resolved_model_url:
        downloaded_path = download_model_file(resolved_model_url, resolved_model_path)
        if resolved_model_sha256:
            verify_sha256(downloaded_path, resolved_model_sha256)
        return downloaded_path

    raise FileNotFoundError(
        f"Model file '{resolved_model_path}' not found. Place the trained model in the project root, set {MODEL_PATH_ENV}, or provide {MODEL_URL_ENV}."
    )