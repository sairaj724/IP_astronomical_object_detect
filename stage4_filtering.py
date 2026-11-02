import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_gaussian_filter(enhanced: np.ndarray,
                           kernel_size: int = 3,
                           sigma: float = 0) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd. Got {kernel_size}.")

    gaussian_out = cv2.GaussianBlur(enhanced,
                                    ksize=(kernel_size, kernel_size),
                                    sigmaX=sigma,
                                    sigmaY=sigma)

    print(f"\n[Method A] Gaussian Filter  (kernel={kernel_size}x{kernel_size}, "
          f"sigma={'auto' if sigma == 0 else sigma})")
    print(f"  Input  : mean = {enhanced.mean():.2f},  std = {enhanced.std():.2f}")
    print(f"  Output : mean = {gaussian_out.mean():.2f},  std = {gaussian_out.std():.2f}")
    _explain_std_change(enhanced.std(), gaussian_out.std())

    return gaussian_out

def apply_median_filter(enhanced: np.ndarray,
                        kernel_size: int = 3) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd. Got {kernel_size}.")

    median_out = cv2.medianBlur(enhanced, ksize=kernel_size)

    print(f"\n[Method B] Median Filter  (kernel={kernel_size}x{kernel_size})")
    print(f"  Input  : mean = {enhanced.mean():.2f},  std = {enhanced.std():.2f}")
    print(f"  Output : mean = {median_out.mean():.2f},  std = {median_out.std():.2f}")
    _explain_std_change(enhanced.std(), median_out.std())

    return median_out


def _explain_std_change(std_before: float, std_after: float) -> None:
    """Print an interpretation of the change in standard deviation."""
    diff_pct = (std_before - std_after) / std_before * 100
    if diff_pct > 15:
        print(f"  [Note] Std dev reduced by {diff_pct:.1f}% — significant smoothing applied.")
    elif diff_pct > 5:
        print(f"  [Note] Std dev reduced by {diff_pct:.1f}% — mild smoothing.")
    else:
        print(f"  [Note] Std dev changed by only {diff_pct:.1f}% — minimal smoothing (expected for 3x3).")

def display_filtering_comparison(enhanced: np.ndarray,
                                 gaussian_out: np.ndarray,
                                 median_out: np.ndarray,
                                 kernel_size: int = 3,
                                 image_path: str = "") -> None:

    fname = os.path.basename(image_path) if image_path else "image"
    h, w  = enhanced.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Stage 4 — Filtering Comparison  ({fname})  |  Kernel: {kernel_size}×{kernel_size}",
        fontsize=13, y=1.01
    )

    panels = [
        (enhanced,    f"No Filtering\n(contrast-enhanced input)"),
        (gaussian_out, f"Method A: Gaussian Filter\n({kernel_size}×{kernel_size} kernel)"),
        (median_out,   f"Method B: Median Filter\n({kernel_size}×{kernel_size} kernel)"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, pad=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    cy, cx  = h // 2, w // 2
    crop_h, crop_w = 200, 200
    r0, r1  = max(0, cy - crop_h // 2), min(h, cy + crop_h // 2)
    c0, c1  = max(0, cx - crop_w // 2), min(w, cx + crop_w // 2)

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle(
        f"Stage 4 — Zoomed Centre Crop ({crop_h}×{crop_w} px)  |  Kernel: {kernel_size}×{kernel_size}",
        fontsize=12, y=1.01
    )

    zoom_panels = [
        (enhanced[r0:r1, c0:c1],     "No Filtering"),
        (gaussian_out[r0:r1, c0:c1], "Gaussian Filter"),
        (median_out[r0:r1, c0:c1],   "Median Filter"),
    ]
    for ax, (crop, title) in zip(axes2, zoom_panels):
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def display_difference_images(enhanced: np.ndarray,
                               gaussian_out: np.ndarray,
                               median_out: np.ndarray) -> None:

    # Compute absolute differences (cast to int16 first to avoid uint8 underflow)
    diff_gauss  = np.abs(enhanced.astype(np.int16) - gaussian_out.astype(np.int16)).astype(np.uint8)
    diff_median = np.abs(enhanced.astype(np.int16) - median_out.astype(np.int16)).astype(np.uint8)

    # Scale differences for visibility (multiply by 5 so small diffs are visible)
    diff_gauss_vis  = np.clip(diff_gauss.astype(np.int32)  * 5, 0, 255).astype(np.uint8)
    diff_median_vis = np.clip(diff_median.astype(np.int32) * 5, 0, 255).astype(np.uint8)

    print(f"\n  Gaussian  difference: max change = {diff_gauss.max()},  "
          f"mean change = {diff_gauss.mean():.3f}")
    print(f"  Median    difference: max change = {diff_median.max()},  "
          f"mean change = {diff_median.mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 4 — Difference Images  (bright = changed pixel)\n"
                 "Pixel values scaled ×5 for visibility", fontsize=12, y=1.02)

    axes[0].imshow(diff_gauss_vis,  cmap="hot", vmin=0, vmax=255)
    axes[0].set_title("|Original − Gaussian|\n(scaled ×5)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(diff_median_vis, cmap="hot", vmin=0, vmax=255)
    axes[1].set_title("|Original − Median|\n(scaled ×5)", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def run_stage4(enhanced: np.ndarray,
               image_path: str = "",
               kernel_size: int = 3) -> dict:

    print("=" * 60)
    print("  STAGE 4: NOISE REDUCTION AND FILTERING")
    print("=" * 60)
    print(f"\n  Input image: {enhanced.shape[0]}×{enhanced.shape[1]} px, "
          f"dtype={enhanced.dtype}")

    # --- Apply both filters
    gaussian_out = apply_gaussian_filter(enhanced, kernel_size=kernel_size)
    median_out   = apply_median_filter(enhanced,   kernel_size=kernel_size)

    # --- Visual comparison (full image + zoom)
    display_filtering_comparison(enhanced, gaussian_out, median_out,
                                 kernel_size=kernel_size,
                                 image_path=image_path)

    print("\n--- Difference images (what each filter removed) ---")
    display_difference_images(enhanced, gaussian_out, median_out)

    print("\n[Stage 4 complete]")
    print("Inspect the three windows and decide which output to use in Stage 5.")
    print("  Options to pass into Stage 5:")
    print("    results['no_filter']  -> unfiltered enhanced image")
    print("    results['gaussian']   -> Gaussian-filtered image")
    print("    results['median']     -> Median-filtered image")
    print("\nNext step -> Stage 5 (segmentation) when ready.\n")

    return {
        "no_filter":   enhanced,
        "gaussian":    gaussian_out,
        "median":      median_out,
        "kernel_size": kernel_size,
    }

if __name__ == "__main__":

    image_path = r"C:\Users\saira\Downloads\ip-req\img1.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    try:
        from stage1_load_inspect import run_stage1
        from stage2_grayscale    import run_stage2
        from stage3_contrast     import run_stage3

        original_image, _  = run_stage1(image_path)
        gray_image         = run_stage2(original_image, image_path)
        enhanced_image     = run_stage3(gray_image, image_path)
    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)

    # Run Stage 4  — change kernel_size to 5 if 3 is not enough
    results = run_stage4(enhanced_image, image_path, kernel_size=3)
