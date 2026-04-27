"""
Prepare a binary image dataset for the deepfake detector.

Expected source layout by default:
    <source>/ai_images
    <source>/real

The script creates disjoint train/validation splits under:
    dataset/train/fake
    dataset/train/real
    dataset/validation/fake
    dataset/validation/real
"""

import argparse
import os
import random
import shutil
import sys

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_SOURCE = os.environ.get(
    "DEEPFAKE_DATASET_SRC",
    r"C:\Users\dhruv\Downloads\archive (1)\my_real_vs_ai_dataset\my_real_vs_ai_dataset",
)

BASE_DST = os.path.dirname(os.path.abspath(__file__))
TRAIN_FAKE_DST = os.path.join(BASE_DST, "dataset", "train", "fake")
TRAIN_REAL_DST = os.path.join(BASE_DST, "dataset", "train", "real")
VAL_FAKE_DST = os.path.join(BASE_DST, "dataset", "validation", "fake")
VAL_REAL_DST = os.path.join(BASE_DST, "dataset", "validation", "real")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/validation folders.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Root folder containing fake and real image folders.")
    parser.add_argument("--fake-subdir", default="ai_images", help="Fake image subfolder name under --source.")
    parser.add_argument("--real-subdir", default="real", help="Real image subfolder name under --source.")
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clear_folder(folder: str) -> None:
    if os.path.exists(folder):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    os.makedirs(folder, exist_ok=True)


def list_images(src_folder: str) -> list[str]:
    if not os.path.isdir(src_folder):
        raise FileNotFoundError(f"Source folder not found: {src_folder}")

    files = [
        name
        for name in os.listdir(src_folder)
        if name.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not files:
        raise ValueError(f"No supported images found in: {src_folder}")
    return files


def split_files(files: list[str], train_count: int, val_count: int, label: str) -> tuple[list[str], list[str]]:
    needed = train_count + val_count
    if len(files) < needed:
        print(f"  [WARN] {label}: requested {needed} images, found {len(files)}.")
        train_count = min(train_count, len(files))
        val_count = max(0, min(val_count, len(files) - train_count))

    chosen = random.sample(files, train_count + val_count)
    return chosen[:train_count], chosen[train_count:]


def copy_images(src_folder: str, dst_folder: str, files: list[str], label: str) -> None:
    for name in files:
        shutil.copy2(os.path.join(src_folder, name), os.path.join(dst_folder, name))
    print(f"  [OK] Copied {len(files)} {label} images -> {dst_folder}")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    fake_src = os.path.join(args.source, args.fake_subdir)
    real_src = os.path.join(args.source, args.real_subdir)

    print("\n[INFO] Preparing dataset folders...")
    for folder in [TRAIN_FAKE_DST, TRAIN_REAL_DST, VAL_FAKE_DST, VAL_REAL_DST]:
        print(f"  Clearing: {folder}")
        clear_folder(folder)

    try:
        fake_files = list_images(fake_src)
        real_files = list_images(real_src)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        print("        Pass --source or set DEEPFAKE_DATASET_SRC to your dataset root.")
        return 1

    fake_train, fake_val = split_files(fake_files, args.train_per_class, args.val_per_class, "fake")
    real_train, real_val = split_files(real_files, args.train_per_class, args.val_per_class, "real")

    print("\n[COPY] Copying fake images...")
    copy_images(fake_src, TRAIN_FAKE_DST, fake_train, "fake-train")
    copy_images(fake_src, VAL_FAKE_DST, fake_val, "fake-val")

    print("\n[COPY] Copying real images...")
    copy_images(real_src, TRAIN_REAL_DST, real_train, "real-train")
    copy_images(real_src, VAL_REAL_DST, real_val, "real-val")

    print("\n[DONE] Dataset ready. Summary:")
    print(f"   Train -> fake: {len(os.listdir(TRAIN_FAKE_DST))} | real: {len(os.listdir(TRAIN_REAL_DST))}")
    print(f"   Val   -> fake: {len(os.listdir(VAL_FAKE_DST))} | real: {len(os.listdir(VAL_REAL_DST))}")
    print("\n[NEXT] Run: python train_ensemble.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
