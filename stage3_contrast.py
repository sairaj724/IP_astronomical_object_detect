import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

def minmax_stretch(gray: np.ndarray) -> np.ndarray:
    I_min = gray.min()
    I_max = gray.max()

    print(f"\n[Method A] Min-Max Contrast Stretching")
    print(f"  I_min = {I_min},  I_max = {I_max}")

    if I_max == I_min:
        print("  [WARNING] I_min == I_max — image is uniform. Returning unchanged.")
        return gray.copy()

    gray_f   = gray.astype(np.float32)
    stretched = (gray_f - I_min) / (I_max - I_min) * 255.0

    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    print(f"  Result  : min = {stretched.min()},  max = {stretched.max()},  "
          f"mean = {stretched.mean():.2f}")

    if I_min >= 1 and I_max == 255:
        print("  [Note] I_min and I_max are almost at full range already.")
        print("         The stretched image will look nearly identical to the input.")
        print("         Percentile stretching (Method B) is more useful for this image.")

    return stretched

def percentile_stretch(gray: np.ndarray,
                       p_low: float = 2.0,
                       p_high: float = 98.0) -> np.ndarray:

    I_low  = np.percentile(gray, p_low)
    I_high = np.percentile(gray, p_high)

    print(f"\n[Method B] Percentile Contrast Stretching  "
          f"(p_low={p_low}%, p_high={p_high}%)")
    print(f"  {p_low}th percentile  = {I_low:.1f}")
    print(f"  {p_high}th percentile = {I_high:.1f}")

    if I_high == I_low:
        print("  [WARNING] I_low == I_high — cannot stretch. Returning unchanged.")
        return gray.copy()

    gray_f    = gray.astype(np.float32)
    stretched = (gray_f - I_low) / (I_high - I_low) * 255.0

    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    print(f"  Result  : min = {stretched.min()},  max = {stretched.max()},  "
          f"mean = {stretched.mean():.2f},  std = {stretched.std():.2f}")

    return stretched


def display_contrast_comparison(gray: np.ndarray,
                                stretched_mm: np.ndarray,
                                stretched_pct: np.ndarray,
                                image_path: str = "") -> None:
 
    fname = os.path.basename(image_path) if image_path else "image"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Stage 3 — Contrast Enhancement Comparison\n({fname})",
                 fontsize=13, y=1.01)

    panels = [
        (gray,          "Original Grayscale\n(no enhancement)",               gray.min(),          gray.max()),
        (stretched_mm,  "Method A: Min-Max Stretch\nI_new = (I-Imin)/(Imax-Imin) × 255", 0, 255),
        (stretched_pct, "Method B: Percentile Stretch\n(2nd – 98th percentile)",           0, 255),
    ]

    for ax, (img, title, vmin, vmax) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, pad=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def display_histograms(gray: np.ndarray,
                       stretched_mm: np.ndarray,
                       stretched_pct: np.ndarray) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("Stage 3 — Intensity Histograms", fontsize=13, y=1.01)

    labels = [
        "Original Grayscale",
        "Method A: Min-Max",
        "Method B: Percentile (2–98%)",
    ]
    images = [gray, stretched_mm, stretched_pct]

    for ax, img, label in zip(axes, images, labels):
        ax.hist(img.ravel(), bins=128, range=(0, 255),
                color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Pixel Intensity (0–255)")
        ax.set_ylabel("Pixel Count")
        ax.set_xlim(0, 255)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_stage3(gray: np.ndarray,
               image_path: str = "",
               p_low: float = 2.0,
               p_high: float = 98.0) -> np.ndarray:

    print("=" * 60)
    print("  STAGE 3: CONTRAST ENHANCEMENT")
    print("=" * 60)

    stretched_mm = minmax_stretch(gray)

    stretched_pct = percentile_stretch(gray, p_low=p_low, p_high=p_high)

    display_contrast_comparison(gray, stretched_mm, stretched_pct, image_path)

    display_histograms(gray, stretched_mm, stretched_pct)

    print("\n[Decision] Returning the percentile-stretched image for Stage 4.")
    print("           Reason: min-max stretch is ineffective when I_max = 255")
    print("           (a single bright star prevents useful stretching).")
    print("\n[Stage 3 complete]")
    print("Next step -> Stage 4 (filtering) when ready.\n")

    return stretched_pct

if __name__ == "__main__":

    image_path = r"C:\Users\saira\Downloads\ip-req\img1.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    try:
        from stage1_load_inspect import run_stage1
        from stage2_grayscale import run_stage2
        original_image, _ = run_stage1(image_path)
        gray_image = run_stage2(original_image, image_path)
    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)

    enhanced_image = run_stage3(gray_image, image_path)
