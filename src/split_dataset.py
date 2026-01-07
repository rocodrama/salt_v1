# make_split.py
# Usage:
#   python make_split.py --image images --out dataset --train 8 --val 2
#
# Output:
#   dataset/
#     train/images/*.png
#     val/images/*.png
#     splits/train.txt
#     splits/val.txt

import argparse
import shutil
from pathlib import Path
import random

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_files_by_stem(folder: Path):
    """Return dict: stem -> Path for image-like files. If duplicate stems exist, keep the first by sorted order."""
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    out = {}
    for p in files:
        if p.stem not in out:
            out[p.stem] = p
    return out


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Image folder")
    ap.add_argument("--out", required=True, help="Output dataset folder")
    ap.add_argument("--train", type=int, default=8, help="Train ratio numerator (default: 8)")
    ap.add_argument("--val", type=int, default=2, help="Val ratio numerator (default: 2)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument("--mode",choices=["copy", "symlink", "move"],default="copy",help="How to place files in dataset")

    args = ap.parse_args()

    image_dir = Path(args.image)
    out_dir = Path(args.out)

    if args.train <= 0 or args.val <= 0:
        print("[ERROR] --train and --val must be positive integers.")
        return

    images = list_files_by_stem(image_dir)
    stems = sorted(images.keys())

    print(f"[INFO] total images: {len(stems)}")

    if len(stems) == 0:
        print("[ERROR] No images found.")
        return

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(stems)

    # Split by ratio
    total = len(stems)
    train_count = int(total * (args.train / (args.train + args.val)))
    train_count = max(1, min(train_count, total - 1)) if total >= 2 else total

    train_stems = stems[:train_count]
    val_stems = stems[train_count:]

    print(f"[INFO] split ratio train:val = {args.train}:{args.val}")
    print(f"[INFO] train={len(train_stems)}  val={len(val_stems)}")
    print(f"[INFO] mode={args.mode}")

    # Create dirs
    train_img_dir = out_dir / "train" / "images"
    val_img_dir = out_dir / "val" / "images"
    ensure_dir(train_img_dir)
    ensure_dir(val_img_dir)

    # Copy/link files
    def place(stems, dst_dir):
        for i, stem in enumerate(stems, 1):
            src = images[stem]
            dst = dst_dir / src.name
            copy_or_link(src, dst, args.mode)

            if i % 200 == 0:
                print(f"[INFO] placed {i}/{len(stems)}")

    place(train_stems, train_img_dir)
    place(val_stems, val_img_dir)

    # Save split lists
    splits_dir = out_dir / "splits"
    ensure_dir(splits_dir)

    with open(splits_dir / "train.txt", "w", encoding="utf-8") as f:
        for s in sorted(train_stems):
            f.write(s + "\n")

    with open(splits_dir / "val.txt", "w", encoding="utf-8") as f:
        for s in sorted(val_stems):
            f.write(s + "\n")

    print("[DONE]")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
