import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_roi_file(roi_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = roi_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in roi_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def to_binary_mask(roi: np.ndarray):
    if roi.ndim == 3:
        roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_g = roi
    return (roi_g > 0).astype(np.uint8)


def largest_square_inside_roi(mask01: np.ndarray):
    """
    Find the largest all-1 square in mask01.
    Returns (side, x1, y1, x2, y2) where (x1,y1) is top-left, (x2,y2) is exclusive.
    If none, returns None.
    """
    h, w = mask01.shape

    ys, xs = np.where(mask01 == 1)
    if xs.size == 0:
        return None

    # centroid for tie-breaking
    cx = float(xs.mean())
    cy = float(ys.mean())

    # dp[y, x] = size of largest square ending at (y-1, x-1) in original coordinates
    # dp has shape (h+1, w+1) to simplify borders
    dp = np.zeros((h + 1, w + 1), dtype=np.int32)

    best_side = 0
    best = None  # (dist2, side, x1, y1, x2, y2)

    for y in range(1, h + 1):
        row = mask01[y - 1]
        dp_row = dp[y]
        dp_prev = dp[y - 1]
        for x in range(1, w + 1):
            if row[x - 1] == 1:
                s = 1 + min(dp_prev[x], dp_row[x - 1], dp_prev[x - 1])
                dp_row[x] = s

                if s > best_side:
                    best_side = s
                    x2 = x
                    y2 = y
                    x1 = x2 - s
                    y1 = y2 - s
                    px = x1 + s / 2.0
                    py = y1 + s / 2.0
                    d2 = (px - cx) ** 2 + (py - cy) ** 2
                    best = (d2, s, x1, y1, x2, y2)

                elif s == best_side and best_side > 0:
                    x2 = x
                    y2 = y
                    x1 = x2 - s
                    y1 = y2 - s
                    px = x1 + s / 2.0
                    py = y1 + s / 2.0
                    d2 = (px - cx) ** 2 + (py - cy) ** 2
                    # choose closer to centroid
                    if best is None or d2 < best[0]:
                        best = (d2, s, x1, y1, x2, y2)

    if best_side == 0 or best is None:
        return None

    _, side, x1, y1, x2, y2 = best
    return int(side), int(x1), int(y1), int(x2), int(y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image folder")
    ap.add_argument("--roi", required=True, help="roi mask folder (same filenames)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--out_size", type=int, default=256, help="output size (default: 256)")
    ap.add_argument("--min_side", type=int, default=32, help="skip if max square side < this (default: 32)")
    args = ap.parse_args()

    image_dir = Path(args.image)
    roi_dir = Path(args.roi)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    img_paths.sort()

    print(f"[INFO] image_dir: {image_dir} ({len(img_paths)} files)")
    print(f"[INFO] roi_dir:   {roi_dir}")
    print(f"[INFO] out_dir:   {out_dir}")
    print(f"[INFO] out_size:  {args.out_size}x{args.out_size}")
    print(f"[INFO] min_side:  {args.min_side}")

    if not img_paths:
        print("[ERROR] No images found. (png/jpg/jpeg/bmp/tif/tiff/webp)")
        return

    ok, skip_no_roi, skip_empty, skip_too_small, fail = 0, 0, 0, 0, 0

    for img_path in img_paths:
        roi_path = find_roi_file(roi_dir, img_path.stem)
        if roi_path is None:
            print(f"[SKIP] ROI not found: {img_path.name}")
            skip_no_roi += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[FAIL] cannot read image: {img_path}")
            fail += 1
            continue

        roi = cv2.imread(str(roi_path), cv2.IMREAD_UNCHANGED)
        if roi is None:
            print(f"[FAIL] cannot read roi: {roi_path}")
            fail += 1
            continue

        mask01 = to_binary_mask(roi)

        # ROI size mismatch -> resize ROI to image using nearest
        if mask01.shape[0] != img.shape[0] or mask01.shape[1] != img.shape[1]:
            mask01 = cv2.resize(mask01, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if mask01.max() == 0:
            print(f"[SKIP] empty ROI: {roi_path.name}")
            skip_empty += 1
            continue

        res = largest_square_inside_roi(mask01)
        if res is None:
            print(f"[SKIP] no square inside ROI: {img_path.name}")
            skip_empty += 1
            continue

        side, x1, y1, x2, y2 = res
        if side < args.min_side:
            print(f"[SKIP] max side too small ({side}px): {img_path.name}")
            skip_too_small += 1
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[FAIL] empty crop: {img_path.name}")
            fail += 1
            continue

        crop_rs = cv2.resize(crop, (args.out_size, args.out_size), interpolation=cv2.INTER_LINEAR)
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), crop_rs)

        ok += 1
        if ok % 50 == 0:
            print(f"[INFO] saved {ok} crops...")

    print("\n=== Done ===")
    print(f"OK: {ok}")
    print(f"SKIP(no roi): {skip_no_roi}")
    print(f"SKIP(empty roi): {skip_empty}")
    print(f"SKIP(too small): {skip_too_small}")
    print(f"FAIL: {fail}")


if __name__ == "__main__":
    main()
