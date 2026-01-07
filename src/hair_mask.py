import os
import argparse
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main(input_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 모델 다운로드
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    MODEL_PATH = "selfie_multiclass_256x256.tflite"

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Saved:", MODEL_PATH)

    # 2) ImageSegmenter 생성
    BaseOptions = python.BaseOptions
    ImageSegmenter = vision.ImageSegmenter
    ImageSegmenterOptions = vision.ImageSegmenterOptions
    RunningMode = vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )

    segmenter = ImageSegmenter.create_from_options(options)

    # 3) png 파일 순회
    png_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ])

    print(f"Found {len(png_files)} png files")

    HAIR_CLASS_ID = 1

    for idx, fname in enumerate(png_files, 1):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(out_dir, fname)

        print(f"[{idx}/{len(png_files)}] Processing: {fname}")

        mp_image = mp.Image.create_from_file(in_path)

        # 4) 추론
        result = segmenter.segment(mp_image)

        category_mask = getattr(result, "category_mask", None) or getattr(result, "categoryMask", None)
        if category_mask is None:
            print(f"  -> Skipped (no category mask): {fname}")
            continue

        mask_np = category_mask.numpy_view()  # (H, W)

        # 5) hair 마스크 추출
        hair_mask = (mask_np == HAIR_CLASS_ID).astype(np.uint8) * 255

        cv2.imwrite(out_path, hair_mask)

        print(f"  -> Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory containing png files")
    parser.add_argument("--out", required=True, help="Output directory")

    args = parser.parse_args()

    main(args.input, args.out)
