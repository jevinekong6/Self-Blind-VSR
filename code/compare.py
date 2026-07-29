import cv2, os, glob
import numpy as np


# ---- EDIT THESE ----
LR_DIR = '/workspace/dataset/MyVideos/LR_blurdown_x4/GX_KyleField'
HR_DIR = '/workspace/infer_results/infer_2026-07-24_KyleField/GX_KyleField'   # SR output
OUT_DIR = '/workspace/infer_results/comparison_GX_KyleField'
# --------------------

os.makedirs(OUT_DIR, exist_ok=True)

lr_files = sorted(glob.glob(os.path.join(LR_DIR, '*.png')))
hr_files = sorted(glob.glob(os.path.join(HR_DIR, '*.png')))

if not lr_files or not hr_files:
    raise SystemExit(f"Empty folder: {len(lr_files)} LR, {len(hr_files)} HR")

for lr_path, hr_path in zip(lr_files, hr_files):
    lr = cv2.imread(lr_path)
    hr = cv2.imread(hr_path)

    # upscale LR to HR height, nearest-neighbor keeps it blocky (honest)
    h, w = hr.shape[:2]
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4px white divider
    divider = 255 * np.ones((h, 4, 3), dtype=lr_up.dtype)
    combined = cv2.hconcat([lr_up, divider, hr])

    cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(hr_path)), combined)

print(f"Wrote {len(hr_files)} comparisons to {OUT_DIR}")