import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def run_connected_components(mask: np.ndarray) -> tuple:
   
    if mask.max() == 1:
        mask_u8 = (mask * 255).astype(np.uint8)
    else:
        mask_u8 = mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    # Label 0 is the background — actual objects start at label 1
    n_objects = num_labels - 1
    print(f"\n[CCL] Total components found (including background): {num_labels}")
    print(f"[CCL] Candidate regions (foreground components)    : {n_objects}")

    return num_labels, labels, stats, centroids

def filter_regions(num_labels: int,
                   stats: np.ndarray,
                   centroids: np.ndarray,
                   image_shape: tuple,
                   min_area: int = 5,
                   max_area: int = None,
                   remove_border: bool = True) -> list:

    H, W = image_shape[:2]
    kept = []
    n_too_small  = 0
    n_too_large  = 0
    n_border     = 0

    for label_id in range(1, num_labels):   # skip label 0 (background)
        x    = int(stats[label_id, cv2.CC_STAT_LEFT])
        y    = int(stats[label_id, cv2.CC_STAT_TOP])
        w    = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h    = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx   = float(centroids[label_id, 0])
        cy   = float(centroids[label_id, 1])

        # Filter 1: Minimum area
        if area < min_area:
            n_too_small += 1
            continue

        # Filter 2: Maximum area
        if max_area is not None and area > max_area:
            n_too_large += 1
            continue

        # Filter 3: Border touching
        if remove_border:
            if x <= 0 or y <= 0 or (x + w) >= W or (y + h) >= H:
                n_border += 1
                continue

        kept.append({
            "id":   label_id,
            "x":    x,
            "y":    y,
            "w":    w,
            "h":    h,
            "area": area,
            "cx":   round(cx, 1),
            "cy":   round(cy, 1),
        })

    print(f"\n--- Region Filtering Report ---")
    print(f"  Total raw regions       : {num_labels - 1}")
    print(f"  Removed (area < {min_area:4d} px): {n_too_small}")
    if max_area:
        print(f"  Removed (area > {max_area:4d} px): {n_too_large}")
    if remove_border:
        print(f"  Removed (touch border)  : {n_border}")
    print(f"  Kept candidate regions  : {len(kept)}")
    print(f"--------------------------------")

    return kept

def measure_intensity(regions: list,
                      labels: np.ndarray,
                      gray: np.ndarray) -> list:

    for region in regions:
        lid  = region["id"]
        # Extract all pixel values that belong to this label
        pixel_values = gray[labels == lid]
        region["mean_intensity"] = round(float(pixel_values.mean()), 2)

    print(f"[OK] Mean intensity measured for {len(regions)} regions.")
    return regions

def display_detected_objects(gray: np.ndarray,
                             regions: list,
                             image_path: str = "",
                             max_display: int = 500) -> None:

    fname  = os.path.basename(image_path) if image_path else "image"
    n_show = min(len(regions), max_display)

    # Sort by area descending to draw the largest first if we must cap
    sorted_regions = sorted(regions, key=lambda r: r["area"], reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Show grayscale image as background
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)

    for i, r in enumerate(sorted_regions[:n_show]):
        # Bounding box rectangle (x, y = top-left corner)
        rect = mpatches.Rectangle(
            (r["x"], r["y"]), r["w"], r["h"],
            linewidth=0.8,
            edgecolor="cyan",
            facecolor="none",
            alpha=0.8
        )
        ax.add_patch(rect)

        # Centroid marker
        ax.plot(r["cx"], r["cy"], "+", color="red",
                markersize=4, markeredgewidth=0.8, alpha=0.8)

    title_str = (f"Stage 6 — Detected Candidate Objects  ({fname})\n"
                 f"Total: {len(regions)} regions  "
                 f"(cyan = bounding box,  red + = centroid)")
    if len(regions) > max_display:
        title_str += f"\n[Note] Only {max_display} largest shown for clarity"

    ax.set_title(title_str, fontsize=11, pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def display_area_distribution(regions: list) -> None:

    if not regions:
        print("[Stage 6] No regions to plot.")
        return

    areas = [r["area"] for r in regions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 6 — Region Area Distribution", fontsize=13)

    # Left: full histogram (log y-scale because area range is wide)
    axes[0].hist(areas, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("Region area (pixels)")
    axes[0].set_ylabel("Number of regions")
    axes[0].set_title("Area Histogram (all regions)")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    # Right: top-10 largest regions
    top10   = sorted(regions, key=lambda r: r["area"], reverse=True)[:10]
    top_ids = [f"ID {r['id']}" for r in top10]
    top_a   = [r["area"] for r in top10]
    axes[1].barh(top_ids[::-1], top_a[::-1], color="coral", alpha=0.85)
    axes[1].set_xlabel("Area (pixels)")
    axes[1].set_title("Top 10 Largest Candidate Regions")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_stage6(mask: np.ndarray,
               gray: np.ndarray,
               image_path: str = "",
               min_area: int = 5,
               max_area: int = None,
               remove_border: bool = True) -> list:

    print("=" * 60)
    print("  STAGE 6: OBJECT IDENTIFICATION AND SEPARATION")
    print("=" * 60)

    # Step A: CCL
    num_labels, labels, stats, centroids = run_connected_components(mask)

    if num_labels <= 1:
        print("[Stage 6] No foreground components found in the mask.")
        print("          Check Stage 5 — the threshold may be too high.")
        return []

    # Step B: Filter
    regions = filter_regions(
        num_labels, stats, centroids,
        image_shape   = mask.shape,
        min_area      = min_area,
        max_area      = max_area,
        remove_border = remove_border
    )

    if not regions:
        print("[Stage 6] All regions were removed by filters.")
        print("          Try lowering min_area or disabling remove_border.")
        return []

    # Step C: Measure intensity
    regions = measure_intensity(regions, labels, gray)

    # Step D: Print summary table (top 15 by area)
    print(f"\n--- Top 15 regions by area ---")
    print(f"  {'ID':>5}  {'Area':>7}  {'CentX':>7}  {'CentY':>7}  "
          f"{'W':>5}  {'H':>5}  {'MeanI':>7}")
    print("  " + "-" * 55)
    for r in sorted(regions, key=lambda x: x["area"], reverse=True)[:15]:
        print(f"  {r['id']:>5}  {r['area']:>7}  {r['cx']:>7.1f}  "
              f"{r['cy']:>7.1f}  {r['w']:>5}  {r['h']:>5}  "
              f"{r['mean_intensity']:>7.1f}")

    # Step E: Visualise
    display_detected_objects(gray, regions, image_path)

    # Step F: Area distribution
    display_area_distribution(regions)

    print(f"\n[Stage 6 complete]")
    print(f"  Total candidate objects identified: {len(regions)}")
    print("Next step -> Stage 7 (feature extraction & catalogue) when ready.\n")

    return regions

if __name__ == "__main__":

    image_path = r"C:\Users\saira\Downloads\ip-req\img1.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    try:
        from stage1_load_inspect import run_stage1
        from stage2_grayscale    import run_stage2
        from stage3_contrast     import run_stage3
        from stage4_filtering    import run_stage4
        from stage5_segmentation import run_stage5

        original_image, _  = run_stage1(image_path)
        gray_image         = run_stage2(original_image, image_path)
        enhanced_image     = run_stage3(gray_image, image_path)
        filter_results     = run_stage4(enhanced_image, image_path, kernel_size=3)
        median_image       = filter_results["median"]
        seg_results        = run_stage5(median_image, image_path,
                                        global_t=130, block_size=51, C=5)
        otsu_mask          = seg_results["mask_otsu"]   # chosen method

    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)

    regions = run_stage6(
        mask          = otsu_mask,
        gray          = enhanced_image,   # use contrast-enhanced for intensity
        image_path    = image_path,
        min_area      = 5,
        max_area      = None,
        remove_border = True,
    )
