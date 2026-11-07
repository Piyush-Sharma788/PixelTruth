"""Download and unpack the 140k Real and Fake Faces dataset for PixelTruth.

Source: the public Hugging Face mirror ``TheKernel01/140k-Real-and-Fake-Faces``
(same images and train/validation/test split as the Kaggle dataset
``xhlulu/140k-real-and-fake-faces``: real faces from FFHQ, fake faces from
StyleGAN).  No account or token is required.

The script writes the original JPEG bytes (no re-encoding) into the folder
layout that ``experiments/train_v4.py`` and ``evaluate.py`` expect::

    data/faces/
      train/{real,fake}/*.jpg
      validation/{real,fake}/*.jpg
      test/{real,fake}/*.jpg

Usage::

    python scripts/prepare_dataset.py                 # 3 train shards (~50k imgs), full val + test
    python scripts/prepare_dataset.py --train-shards 6  # full 100k training set
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF_PARQUET_BASE = (
    "https://huggingface.co/api/datasets/TheKernel01/140k-Real-and-Fake-Faces"
    "/parquet/default"
)
SHARDS = {"train": 6, "validation": 2, "test": 2}
# The HF dataset's ClassLabel names are ["real", "fake"] (0 = real, 1 = fake).
# We write folders by *name*; Keras/evaluate.py then assign fake = 0, real = 1
# alphabetically, which is the convention the rest of PixelTruth uses.
HF_LABEL_NAMES = {0: "real", 1: "fake"}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetch   {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)
    return dest


def unpack(parquet_path: Path, split_dir: Path, prefix: str) -> dict[str, int]:
    import pyarrow.parquet as pq

    counts = {"real": 0, "fake": 0}
    pf = pq.ParquetFile(parquet_path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["image", "label"])
        for row in table.to_pylist():
            name = HF_LABEL_NAMES[int(row["label"])]
            out_dir = split_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = counts[name]
            (out_dir / f"{prefix}_{idx:06d}.jpg").write_bytes(row["image"]["bytes"])
            counts[name] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/faces", help="output root (default: data/faces)")
    parser.add_argument("--cache", default="data/hf_140k", help="parquet download cache")
    parser.add_argument("--train-shards", type=int, default=3, help="train shards to use (1-6, ~16.7k images each)")
    parser.add_argument("--val-shards", type=int, default=1, help="validation shards (1-2, 10k images each)")
    parser.add_argument("--test-shards", type=int, default=2, help="test shards (1-2, 10k images each)")
    args = parser.parse_args(argv)

    wanted = {"train": args.train_shards, "validation": args.val_shards, "test": args.test_shards}
    out_root = Path(args.out)
    cache = Path(args.cache)

    for split, n in wanted.items():
        n = max(0, min(n, SHARDS[split]))
        split_dir = out_root / split
        totals = {"real": 0, "fake": 0}
        for i in range(n):
            pq_path = download(f"{HF_PARQUET_BASE}/{split}/{i}.parquet", cache / f"{split}_{i}.parquet")
            counts = unpack(pq_path, split_dir, prefix=f"{split}{i}")
            for k in totals:
                totals[k] += counts[k]
        print(f"{split:<10} real={totals['real']:>6} fake={totals['fake']:>6} -> {split_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
