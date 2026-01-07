import argparse
import os
import numpy as np
import cv2


def _ri(rng, a, b):
    return int(rng.integers(a, b + 1))


def draw_blob(mask, rng, n=3):
    h, w = mask.shape
    for _ in range(n):
        cx = _ri(rng, 0, w - 1)
        cy = _ri(rng, 0, h - 1)
        rx = _ri(rng, w // 30, w // 10)   # 작은 덩어리
        ry = _ri(rng, h // 30, h // 10)
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1, -1)


def draw_stroke(mask, rng, n=2):
    h, w = mask.shape
    for _ in range(n):
        k = _ri(rng, 3, 6)
        pts = []
        x = _ri(rng, w // 6, w - w // 6)
        y = _ri(rng, h // 6, h - h // 6)
        pts.append([x, y])
        for _i in range(1, k):
            x = np.clip(x + _ri(rng, -w // 8, w // 8), 0, w - 1)
            y = np.clip(y + _ri(rng, -h // 8, h // 8), 0, h - 1)
            pts.append([int(x), int(y)])
        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        thickness = _ri(rng, 2, 6)
        cv2.polylines(mask, [pts], False, 1, thickness)


def draw_islands(mask, rng, n=30):
    h, w = mask.shape
    for _ in range(n):
        cx = _ri(rng, 0, w - 1)
        cy = _ri(rng, 0, h - 1)
        r = _ri(rng, 1, 3)
        cv2.circle(mask, (cx, cy), r, 1, -1)


def make_binary_island_mask_256(rng, cover_range=(0.03, 0.18)):
    h = w = 256
    lo, hi = cover_range

    best = None
    best_dist = 1e9

    for _ in range(40):
        mask = np.zeros((h, w), dtype=np.uint8)

        if rng.random() < 0.8:
            draw_blob(mask, rng, _ri(rng, 1, 4))
        if rng.random() < 0.6:
            draw_stroke(mask, rng, _ri(rng, 1, 3))
        if rng.random() < 0.7:
            draw_islands(mask, rng, _ri(rng, 10, 60))

        # 약간만 morphology (너무 연결되지 않게!)
        if rng.random() < 0.5:
            k = _ri(rng, 3, 7)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cover = mask.mean()

        if lo <= cover <= hi:
            return mask

        d = min(abs(cover - lo), abs(cover - hi))
        if d < best_dist:
            best_dist = d
            best = mask

    return best if best is not None else np.zeros((256, 256), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[INFO] generating {args.num} binary island masks (0/1)")

    for i in range(args.num):
        mask = make_binary_island_mask_256(rng)
        out_path = os.path.join(args.out, f"mask_{i:05d}.png")
        cv2.imwrite(out_path, mask * 255)  # 저장은 보기 편하게 0/255
        if (i + 1) % 100 == 0:
            print(f"[INFO] {i + 1}/{args.num}")

    print("[DONE]")


if __name__ == "__main__":
    main()
