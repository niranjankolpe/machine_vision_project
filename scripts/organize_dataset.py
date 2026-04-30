"""
organize_dataset.py
--------------------
Organizes the raw WhatsApp cotton images into class folders.
The 5 time-groups map to 5 cotton disease classes.

Usage:
    python scripts/organize_dataset.py --src path/to/raw_images
"""

import os
import re
import shutil
import random
import argparse
from pathlib import Path

# ── Disease class mapping ─────────────────────────────────────────────────────
# Mapped from WhatsApp image timestamps (each time group = one photo session)
# You can rename these to match your actual labels if needed.
DISEASE_CLASSES = [
    "Bacterial_Blight",
    "Healthy",
    "Alternaria_Leaf_Spot",
    "Curl_Virus",
    "Fusarium_Wilt",
]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def get_time_group(filename: str) -> str | None:
    m = re.search(r"at (\d+\.\d+)\.\d+", filename)
    return m.group(1) if m else None


def organize(src_dir: str, dst_dir: str = "dataset"):
    src = Path(src_dir)
    images = sorted(
        [f for f in src.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    # Group images by minute-level timestamp
    groups: dict[str, list[Path]] = {}
    for img in images:
        key = get_time_group(img.name)
        if key:
            groups.setdefault(key, []).append(img)

    sorted_keys = sorted(groups.keys())
    print(f"\n✅ Found {len(images)} images in {len(sorted_keys)} groups\n")

    if len(sorted_keys) > len(DISEASE_CLASSES):
        print(
            f"⚠️  More groups ({len(sorted_keys)}) than classes ({len(DISEASE_CLASSES)}). "
            "Extra groups will be ignored. Edit DISEASE_CLASSES to add more."
        )

    for split in ["train", "val", "test"]:
        for cls in DISEASE_CLASSES:
            (Path(dst_dir) / split / cls).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for i, key in enumerate(sorted_keys):
        if i >= len(DISEASE_CLASSES):
            break
        cls_name = DISEASE_CLASSES[i]
        files = groups[key]
        random.shuffle(files)

        n = len(files)
        n_train = max(1, int(n * SPLIT_RATIOS["train"]))
        n_val   = max(1, int(n * SPLIT_RATIOS["val"]))

        splits = {
            "train": files[:n_train],
            "val":   files[n_train : n_train + n_val],
            "test":  files[n_train + n_val :],
        }
        # Guarantee at least 1 image in val/test
        for split in ["val", "test"]:
            if not splits[split]:
                splits[split] = [splits["train"].pop()]

        for split, split_files in splits.items():
            for j, f in enumerate(split_files):
                dst = Path(dst_dir) / split / cls_name / f"{cls_name}_{j:04d}{f.suffix}"
                shutil.copy2(f, dst)
                total_copied += 1

        print(
            f"  [{i+1}/{len(DISEASE_CLASSES)}] {cls_name:25s} "
            f"→ {len(splits['train'])} train | {len(splits['val'])} val | {len(splits['test'])} test"
        )

    print(f"\n✅ Done. {total_copied} images copied to '{dst_dir}/'")
    print("\nDataset structure:")
    for split in ["train", "val", "test"]:
        for cls in DISEASE_CLASSES:
            p = Path(dst_dir) / split / cls
            n = len(list(p.glob("*")))
            if n:
                print(f"  {split}/{cls}: {n} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", default="raw_images", help="Folder with raw WhatsApp images"
    )
    parser.add_argument(
        "--dst", default="dataset", help="Output dataset root folder"
    )
    args = parser.parse_args()
    organize(args.src, args.dst)
