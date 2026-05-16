import argparse
import os

from model_utils import (
    DEFAULT_MODEL_PATH,
    MODEL_PATH_ENV,
    MODEL_SHA256_ENV,
    MODEL_URL_ENV,
    ensure_model_file,
    get_model_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download the PixelTruth trained model.")
    parser.add_argument("--url", default=os.getenv(MODEL_URL_ENV, ""), help="Direct URL to the model file.")
    parser.add_argument(
        "--sha256",
        default=os.getenv(MODEL_SHA256_ENV, ""),
        help="Optional SHA256 checksum for the model file.",
    )
    parser.add_argument(
        "--dest",
        default=get_model_path(),
        help=f"Destination path for the model file. Defaults to {DEFAULT_MODEL_PATH} or ${MODEL_PATH_ENV}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_model_file(
        model_path=args.dest,
        model_url=args.url,
        model_sha256=args.sha256,
        download_if_missing=True,
    )
    print(f"Model saved to {args.dest}")


if __name__ == "__main__":
    main()