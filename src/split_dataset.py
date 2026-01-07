# make_split.py
# Usage:
#   python make_split.py --image crop --mask masks --out dataset --train 8 --val 2
#
# Output:
#   dataset/
#     train/images/*.png
#     train/masks/*.png
#     val/images/*.png
#     val/masks/*.png

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
        # Windows needs admin/dev mode for symlinks; fallback to copy on failure
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Image folder")
    ap.add_argument("--mask", required=True, help="Mask folder (same filenames/stems)")
    ap.add_argument("--out", required=True, help="Output dataset folder")
    ap.add_argument("--train", type=int, default=8, help="Train ratio numerator (default: 8)")
    ap.add_argument("--val", type=int, default=2, help="Val ratio numerator (default: 2)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="How to place files in dataset")
    args = ap.parse_args()

    image_dir = Path(args.image)
    mask_dir = Path(args.mask)
    out_dir = Path(args.out)

    if args.train <= 0 or args.val <= 0:
        print("[ERROR] --train and --val must be positive integers.")
        return

    images = list_files_by_stem(image_dir)
    masks = list_files_by_stem(mask_dir)

    common_stems = sorted(set(images.keys()) & set(masks.keys()))
    only_img = sorted(set(images.keys()) - set(masks.keys()))
    only_msk = sorted(set(masks.keys()) - set(images.keys()))

    print(f"[INFO] images: {len(images)}  masks: {len(masks)}")
    print(f"[INFO] matched pairs: {len(common_stems)}")
    if only_img:
        print(f"[INFO] image-only (no matching mask): {len(only_img)} (e.g. {only_img[:5]})")
    if only_msk:
        print(f"[INFO] mask-only (no matching image): {len(only_msk)} (e.g. {only_msk[:5]})")

    if len(common_stems) == 0:
        print("[ERROR] No matched pairs found (same stem).")
        return

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(common_stems)

    # Split by ratio
    total = len(common_stems)
    train_count = int(total * (args.train / (args.train + args.val)))
    train_count = max(1, min(train_count, total - 1)) if total >= 2 else total
    val_count = total - train_count

    train_stems = common_stems[:train_count]
    val_stems = common_stems[train_count:]

    print(f"[INFO] split ratio train:val = {args.train}:{args.val}")
    print(f"[INFO] train={len(train_stems)}  val={len(val_stems)}")
    print(f"[INFO] mode={args.mode}")

    # Create dirs
    train_img_dir = out_dir / "train" / "images"
    train_msk_dir = out_dir / "train" / "masks"
    val_img_dir = out_dir / "val" / "images"
    val_msk_dir = out_dir / "val" / "masks"
    for d in [train_img_dir, train_msk_dir, val_img_dir, val_msk_dir]:
        ensure_dir(d)

    # Copy/link files, keep original filenames (not just stem)
    def place(stems, img_dst, msk_dst):
        for i, stem in enumerate(stems, 1):
            img_src = images[stem]
            msk_src = masks[stem]
            img_out = img_dst / img_src.name
            msk_out = msk_dst / msk_src.name
            copy_or_link(img_src, img_out, args.mode)
            copy_or_link(msk_src, msk_out, args.mode)

            if i % 200 == 0:
                print(f"[INFO] placed {i}/{len(stems)}")

    place(train_stems, train_img_dir, train_msk_dir)
    place(val_stems, val_img_dir, val_msk_dir)

    # Save split lists
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "splits" / "train.txt", "w", encoding="utf-8") as f:
        for s in sorted(train_stems):
            f.write(s + "\n")
    with open(out_dir / "splits" / "val.txt", "w", encoding="utf-8") as f:
        for s in sorted(val_stems):
            f.write(s + "\n")

    print("[DONE]")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
