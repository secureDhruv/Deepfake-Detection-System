"""
prepare_dataset.py
------------------
Copies images from the downloaded dataset into the correct folder structure
for train_model.py.
"""

import os
import sys
import shutil
import random

# Force stdout to be utf-8 immune or just use pure ascii in prints
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_PER_CLASS  = 5000   
VAL_PER_CLASS    = 1000   
SEED = 42  

BASE_SRC   = r"C:\Users\dhruv\Downloads\archive (1)\my_real_vs_ai_dataset\my_real_vs_ai_dataset"
FAKE_SRC   = os.path.join(BASE_SRC, "ai_images")
REAL_SRC   = os.path.join(BASE_SRC, "real")

BASE_DST        = os.path.dirname(os.path.abspath(__file__))
TRAIN_FAKE_DST  = os.path.join(BASE_DST, "dataset", "train", "fake")
TRAIN_REAL_DST  = os.path.join(BASE_DST, "dataset", "train", "real")
VAL_FAKE_DST    = os.path.join(BASE_DST, "dataset", "validation", "fake")
VAL_REAL_DST    = os.path.join(BASE_DST, "dataset", "validation", "real")

def clear_folder(folder: str):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            fp = os.path.join(folder, f)
            if os.path.isfile(fp):
                os.remove(fp)
    os.makedirs(folder, exist_ok=True)

def copy_images(src_folder: str, dst_folder: str, n: int, label: str):
    all_files = [f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    if len(all_files) < n:
        print(f"  [WARN] Only {len(all_files)} images in {src_folder}, using all of them.")
        n = len(all_files)

    chosen = random.sample(all_files, n)
    for fname in chosen:
        shutil.copy2(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))
    print(f"  [OK] Copied {n} {label} images -> {dst_folder}")

def main():
    random.seed(SEED)
    print("\n[INFO] Preparing dataset folders...")
    for folder in [TRAIN_FAKE_DST, TRAIN_REAL_DST, VAL_FAKE_DST, VAL_REAL_DST]:
        print(f"  Clearing: {folder}")
        clear_folder(folder)

    print("\n[COPY] Copying FAKE images (from ai_images/)...")
    copy_images(FAKE_SRC, TRAIN_FAKE_DST, TRAIN_PER_CLASS, "fake-train")
    copy_images(FAKE_SRC, VAL_FAKE_DST,   VAL_PER_CLASS,   "fake-val")

    print("\n[COPY] Copying REAL images (from real/)...")
    copy_images(REAL_SRC, TRAIN_REAL_DST, TRAIN_PER_CLASS, "real-train")
    copy_images(REAL_SRC, VAL_REAL_DST,   VAL_PER_CLASS,   "real-val")

    print("\n[DONE] Dataset ready! Summary:")
    print(f"   Train  -> fake: {len(os.listdir(TRAIN_FAKE_DST))} | real: {len(os.listdir(TRAIN_REAL_DST))}")
    print(f"   Val    -> fake: {len(os.listdir(VAL_FAKE_DST))} | real: {len(os.listdir(VAL_REAL_DST))}")
    print("\n[NEXT] Now run: python train_model.py")

if __name__ == "__main__":
    main()
