import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _plot_histogram_with_threshold(ax, image: np.ndarray,
                                   threshold: int, title: str) -> None:
    """Plot the pixel intensity histogram with a vertical line at the threshold."""
    ax.hist(image.ravel(), bins=128, range=(0, 255),
            color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(x=threshold, color="red", linewidth=1.5,
               label=f"T = {threshold}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel count")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 255)
    ax.grid(axis="y", alpha=0.3)

def global_threshold(filtered: np.ndarray,
                     threshold: int = 130) -> np.ndarray:
    _, mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)

    n_fg = int(np.sum(mask == 255))
    n_bg = int(np.sum(mask == 0))
    total = filtered.size

    print(f"\n[Method A] Global Thresholding  (T = {threshold})")
    print(f"  Foreground pixels : {n_fg:,}  ({n_fg/total*100:.2f}%)")
    print(f"  Background pixels : {n_bg:,}  ({n_bg/total*100:.2f}%)")
    print(f"  Tip: if foreground% is very high, raise T. If very low, lower T.")

    return mask

def otsu_threshold(filtered: np.ndarray) -> tuple[np.ndarray, int]:
   
    otsu_t, mask = cv2.threshold(filtered, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_t = int(otsu_t)

    n_fg  = int(np.sum(mask == 255))
    n_bg  = int(np.sum(mask == 0))
    total = filtered.size

    print(f"\n[Method B] Otsu's Thresholding  (T = {otsu_t}  — auto-selected)")
    print(f"  Foreground pixels : {n_fg:,}  ({n_fg/total*100:.2f}%)")
    print(f"  Background pixels : {n_bg:,}  ({n_bg/total*100:.2f}%)")

    # Warn if Otsu threshold seems too low for a dark-sky image
    if otsu_t < 50:
        print(f"  [!] Otsu chose a LOW threshold ({otsu_t}).")
        print(f"      This often happens in dark-sky images where the histogram")
        print(f"      is dominated by one background peak. Expect many false")
        print(f"      foreground detections. Compare visually with Method A.")
    elif otsu_t > 180:
        print(f"  [!] Otsu chose a HIGH threshold ({otsu_t}).")
        print(f"      Faint objects may be missed. Compare with Method A.")

    return mask, otsu_t

def adaptive_threshold(filtered: np.ndarray,
                       block_size: int = 51,
                       C: int = 5) -> np.ndarray:

    if block_size % 2 == 0:
        block_size += 1   # ensure odd

    mask = cv2.adaptiveThreshold(
        filtered,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )

    n_fg  = int(np.sum(mask == 255))
    n_bg  = int(np.sum(mask == 0))
    total = filtered.size

    print(f"\n[Method C] Adaptive Thresholding  "
          f"(block_size={block_size}, C={C})")
    print(f"  Foreground pixels : {n_fg:,}  ({n_fg/total*100:.2f}%)")
    print(f"  Background pixels : {n_bg:,}  ({n_bg/total*100:.2f}%)")

    if n_fg / total > 0.40:
        print(f"  [!] Over 40% of pixels classified as foreground.")
        print(f"      Adaptive thresholding may be picking up background texture.")
        print(f"      Consider increasing C or block_size, or using Method A/B instead.")

    return mask

def overlay_mask(original_gray: np.ndarray,
                 mask: np.ndarray,
                 colour: tuple = (0, 255, 0)) -> np.ndarray:

    # Convert grayscale to BGR so we can draw coloured highlights
    overlay = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    # Apply colour tint where mask is 255 (foreground)
    overlay[mask == 255] = colour
    return overlay

def display_segmentation_comparison(filtered: np.ndarray,
                                    mask_global: np.ndarray,
                                    mask_otsu: np.ndarray,
                                    mask_adaptive: np.ndarray,
                                    global_t: int,
                                    otsu_t: int,
                                    block_size: int,
                                    C: int,
                                    image_path: str = "") -> None:

    fname = os.path.basename(image_path) if image_path else "image"

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        f"Stage 5 — Segmentation Comparison  ({fname})",
        fontsize=14, y=1.01
    )

    methods = [
        (mask_global,   f"Method A: Global\n(T = {global_t})"),
        (mask_otsu,     f"Method B: Otsu\n(T = {otsu_t}, auto)"),
        (mask_adaptive, f"Method C: Adaptive\n(block={block_size}, C={C})"),
    ]
    thresholds = [global_t, otsu_t, None]

    # Row 0: Histograms
    for col, (mask, title), t in zip(range(3), methods, thresholds):
        ax = axes[0][col]
        if t is not None:
            _plot_histogram_with_threshold(ax, filtered, t,
                                           f"Histogram — {title.split(chr(10))[0]}")
        else:
            # Adaptive: no single threshold to show; show histogram only
            ax.hist(filtered.ravel(), bins=128, range=(0, 255),
                    color="steelblue", edgecolor="none", alpha=0.8)
            ax.set_title(f"Histogram — {title.split(chr(10))[0]}", fontsize=9)
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Pixel count")
            ax.text(0.6, 0.85, "Local threshold\n(no single T)",
                    transform=ax.transAxes, color="red", fontsize=8)
            ax.set_xlim(0, 255)
            ax.grid(axis="y", alpha=0.3)

    # Row 1: Binary masks
    for col, (mask, title) in enumerate(methods):
        ax = axes[1][col]
        ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Binary Mask\n{title}", fontsize=9)
        ax.axis("off")

    # Row 2: Overlay (green = foreground)
    for col, (mask, title) in enumerate(methods):
        ax = axes[2][col]
        overlay = overlay_mask(filtered, mask)
        rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_overlay)
        ax.set_title(f"Overlay (green = foreground)\n{title}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def display_mask_summary(mask_global: np.ndarray,
                         mask_otsu: np.ndarray,
                         mask_adaptive: np.ndarray,
                         global_t: int,
                         otsu_t: int,
                         block_size: int,
                         C: int) -> None:
    """Show just the three masks side by side for a quick comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Stage 5 — Binary Mask Summary", fontsize=13)

    masks   = [mask_global, mask_otsu, mask_adaptive]
    titles  = [
        f"Method A: Global\nT = {global_t}",
        f"Method B: Otsu\nT = {otsu_t} (auto)",
        f"Method C: Adaptive\nblock={block_size}, C={C}",
    ]
    for ax, mask, title in zip(axes, masks, titles):
        pct = np.sum(mask == 255) / mask.size * 100
        ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{title}\nFG: {pct:.1f}% of pixels", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def run_stage5(filtered: np.ndarray,
               image_path: str = "",
               global_t: int = 130,
               p_low_otsu: float = None,
               block_size: int = 51,
               C: int = 5) -> dict:

    print("=" * 60)
    print("  STAGE 5: IMAGE SEGMENTATION (THRESHOLDING)")
    print("=" * 60)
    print(f"\n  Input image : {filtered.shape[0]}×{filtered.shape[1]} px")
    print(f"  Input stats : mean={filtered.mean():.1f}, "
          f"std={filtered.std():.1f}, "
          f"max={filtered.max()}")

    # Print a quick guide for choosing global threshold
    suggested_t = int(filtered.mean() + 2 * filtered.std())
    suggested_t = min(suggested_t, 240)
    print(f"\n  Suggested starting threshold for Method A:")
    print(f"    mean + 2*std = {filtered.mean():.1f} + 2×{filtered.std():.1f}"
          f" = {suggested_t}")
    print(f"  (This captures pixels that are significantly brighter than average.)")
    print(f"  (Adjust global_t in run_stage5() if needed.)\n")

    # --- Run all three methods
    mask_global   = global_threshold(filtered, threshold=global_t)
    mask_otsu, otsu_t = otsu_threshold(filtered)
    mask_adaptive = adaptive_threshold(filtered, block_size=block_size, C=C)

    # --- Detailed 3×3 comparison grid
    display_segmentation_comparison(
        filtered, mask_global, mask_otsu, mask_adaptive,
        global_t=global_t, otsu_t=otsu_t,
        block_size=block_size, C=C,
        image_path=image_path
    )

    # --- Simple mask summary
    display_mask_summary(mask_global, mask_otsu, mask_adaptive,
                         global_t=global_t, otsu_t=otsu_t,
                         block_size=block_size, C=C)

    print("\n[Stage 5 complete]")
    print("Inspect the masks and overlays, then decide which mask to use:")
    print("  results['mask_global']   -> Method A")
    print("  results['mask_otsu']     -> Method B (usually best for this image)")
    print("  results['mask_adaptive'] -> Method C (check for false detections)")
    print("\nNext step -> Stage 6 (object identification) when ready.\n")

    return {
        "mask_global":   mask_global,
        "mask_otsu":     mask_otsu,
        "mask_adaptive": mask_adaptive,
        "global_t":      global_t,
        "otsu_t":        otsu_t,
        "filtered":      filtered,
    }

if __name__ == "__main__":

    image_path = r"C:\Users\saira\Downloads\ip-req\img1.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    try:
        from stage1_load_inspect import run_stage1
        from stage2_grayscale    import run_stage2
        from stage3_contrast     import run_stage3
        from stage4_filtering    import run_stage4

        original_image, _ = run_stage1(image_path)
        gray_image        = run_stage2(original_image, image_path)
        enhanced_image    = run_stage3(gray_image, image_path)
        filter_results    = run_stage4(enhanced_image, image_path, kernel_size=3)
        median_image      = filter_results["median"]   # chosen in Stage 4
    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)

    seg_results = run_stage5(
        median_image,
        image_path  = image_path,
        global_t    = 130,   # <-- tune this
        block_size  = 51,    # <-- tune this for Method C
        C           = 5,     # <-- tune this for Method C
    )
