import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from stage1_load_inspect import load_image, inspect_image
from stage2_grayscale import convert_to_grayscale
from stage3_contrast import percentile_stretch
from stage4_filtering import apply_median_filter, apply_gaussian_filter
from stage5_segmentation import otsu_threshold, global_threshold
from stage6_objects import run_connected_components, filter_regions, measure_intensity
from stage7_catalogue import build_catalogue, export_catalogue


def create_pipeline_comparison_figure(
    original_rgb: np.ndarray,
    gray: np.ndarray,
    enhanced: np.ndarray,
    filtered: np.ndarray,
    mask: np.ndarray,
    regions: list,
    output_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate a 2x3 comprehensive multi-stage comparison figure:
        [1] Original Image (RGB or Gray)
        [2] Grayscale Luminance
        [3] Contrast-Enhanced (Percentile Stretch)
        [4] Filtered Image (Noise Suppressed)
        [5] Binary Segmentation Mask
        [6] Candidate Objects Overlay (Bounding Boxes & Centroids)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor("#0b0e14")

    titles = [
        "1. Raw Input Image",
        "2. Grayscale Conversion",
        "3. Contrast Enhanced (Percentile)",
        "4. Filtered (Noise Reduced)",
        "5. Binary Mask (Segmentation)",
        f"6. Detected Objects ({len(regions)} Candidates)"
    ]

    # Panel 1: Original
    ax1 = axes[0, 0]
    if original_rgb.ndim == 3:
        ax1.imshow(cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(original_rgb, cmap="gray")
    ax1.set_title(titles[0], color="#64b5f6", fontsize=12, fontweight="bold", pad=10)

    # Panel 2: Grayscale
    ax2 = axes[0, 1]
    ax2.imshow(gray, cmap="gray")
    ax2.set_title(titles[1], color="#64b5f6", fontsize=12, fontweight="bold", pad=10)

    # Panel 3: Enhanced
    ax3 = axes[0, 2]
    ax3.imshow(enhanced, cmap="magma")
    ax3.set_title(titles[2], color="#ffb74d", fontsize=12, fontweight="bold", pad=10)

    # Panel 4: Filtered
    ax4 = axes[1, 0]
    ax4.imshow(filtered, cmap="gray")
    ax4.set_title(titles[3], color="#81c784", fontsize=12, fontweight="bold", pad=10)

    # Panel 5: Binary Mask
    ax5 = axes[1, 1]
    ax5.imshow(mask, cmap="bone")
    ax5.set_title(titles[4], color="#4dd0e1", fontsize=12, fontweight="bold", pad=10)

    # Panel 6: Candidate Objects Overlay
    ax6 = axes[1, 2]
    ax6.imshow(enhanced, cmap="gray")
    ax6.set_title(titles[5], color="#ff80ab", fontsize=12, fontweight="bold", pad=10)

    # Draw bounding boxes and centroids (limit visual clutter to top 300 if dense)
    display_regions = sorted(regions, key=lambda r: r.get("area", 0), reverse=True)[:300]
    for r in display_regions:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cx, cy = r["cx"], r["cy"]
        
        # Color code by area: large = cyan, medium = gold, small = red/magenta
        if r["area"] > 500:
            box_color = "#00e5ff"
        elif r["area"] > 50:
            box_color = "#ffd600"
        else:
            box_color = "#ff4081"

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=0.8,
            edgecolor=box_color,
            facecolor="none",
            alpha=0.85
        )
        ax6.add_patch(rect)
        ax6.plot(cx, cy, "+", color="#ffffff", markersize=3, markeredgewidth=0.7)

    # Standardize axes styling
    for ax in axes.flat:
        ax.set_facecolor("#121722")
        ax.tick_params(colors="#90caf9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2a3b5c")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"[Stage 8] Saved pipeline comparison figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def run_stage8(image_path: str, output_dir: str = "outputs") -> plt.Figure:
    """Run full Stage 1 to Stage 7 pipeline and create the Stage 8 comparison figure."""
    print("=" * 60)
    print(" STAGE 8: PIPELINE VISUALISATION AND COMPARATIVE ANALYSIS ")
    print("=" * 60)

    img = load_image(image_path)
    
    gray, _ = convert_to_grayscale(img)
    
    enhanced = percentile_stretch(gray, p_low=2.0, p_high=98.0)
   
    filtered = apply_median_filter(enhanced, kernel_size=3)
    
    mask, t_val = otsu_threshold(filtered)

    num_labels, labels, stats, centroids = run_connected_components(mask)
    regions = filter_regions(num_labels, stats, centroids, img.shape[:2], min_area=5, remove_border=True)
    regions = measure_intensity(regions, labels, enhanced)
   
    df = build_catalogue(regions)
    csv_path = os.path.join(output_dir, "object_catalogue.csv")
    export_catalogue(df, csv_path)

    fig_path = os.path.join(output_dir, "stage8_pipeline_comparison.png")
    fig = create_pipeline_comparison_figure(img, gray, enhanced, filtered, mask, regions, output_path=fig_path, show=False)
    print("[Stage 8 Complete]")
    return fig


if __name__ == "__main__":
    test_img = r"C:\Users\saira\Downloads\ip-req\img1.png"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    run_stage8(test_img)
