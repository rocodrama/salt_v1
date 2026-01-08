# src/split_dataset.py
#
# Split dataset by matching image/mask filenames (stem-based).
#
# Example:
#   python split_dataset.py \
#     --image crop_images \
#     --mask masks \
#     --train train \
#     --val val \
#     --train_ratio 8 \
#     --val_ratio 2

import argparse
import random
import shutil
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_by_stem(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    out = {}
    for p in files:
        if p.stem not in out:
            out[p.stem] = p
    return out


def copy_pair(img_src: Path, msk_src: Path, img_dst: Path, msk_dst: Path):
    shutil.copy2(img_src, img_dst / img_src.name)
    shutil.copy2(msk_src, msk_dst / msk_src.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image folder")
    ap.add_argument("--mask", required=True, help="mask folder")
    ap.add_argument("--train", required=True, help="train output folder")
    ap.add_argument("--val", required=True, help="val output folder")

    ap.add_argument("--train_ratio", type=int, default=8, help="train ratio numerator")
    ap.add_argument("--val_ratio", type=int, default=2, help="val ratio numerator")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    img_dir = Path(args.image)
    msk_dir = Path(args.mask)
    train_dir = Path(args.train)
    val_dir = Path(args.val)

    img_map = list_by_stem(img_dir)
    msk_map = list_by_stem(msk_dir)

    stems = sorted(set(img_map.keys()) & set(msk_map.keys()))
    if len(stems) == 0:
        print("[ERROR] No matching image-mask pairs found.")
        return

    print(f"[INFO] matched pairs: {len(stems)}")

    random.seed(args.seed)
    random.shuffle(stems)

    total = len(stems)
    train_count = int(total * args.train_ratio / (args.train_ratio + args.val_ratio))
    train_count = max(1, min(train_count, total - 1))

    train_stems = stems[:train_count]
    val_stems = stems[train_count:]

    print(f"[INFO] split: train={len(train_stems)} val={len(val_stems)}")

    # create folders
    for d in [
        train_dir / "images", train_dir / "masks",
        val_dir / "images", val_dir / "masks",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # copy files
    for i, stem in enumerate(train_stems, 1):
        copy_pair(
            img_map[stem],
            msk_map[stem],
            train_dir / "images",
            train_dir / "masks",
        )
        if i % 200 == 0:
            print(f"[TRAIN] copied {i}/{len(train_stems)}")

    for i, stem in enumerate(val_stems, 1):
        copy_pair(
            img_map[stem],
            msk_map[stem],
            val_dir / "images",
            val_dir / "masks",
        )
        if i % 200 == 0:
            print(f"[VAL] copied {i}/{len(val_stems)}")

    # save split list (디버깅/재현용)
    split_dir = train_dir.parent / "splits"
    split_dir.mkdir(exist_ok=True)

    with open(split_dir / "train.txt", "w", encoding="utf-8") as f:
        for s in train_stems:
            f.write(s + "\n")

    with open(split_dir / "val.txt", "w", encoding="utf-8") as f:
        for s in val_stems:
            f.write(s + "\n")

    print("[DONE]")


if __name__ == "__main__":
    main()
