import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

def build_catalogue(regions: list) -> pd.DataFrame:

    if not regions:
        print("[Stage 7] No regions provided. Cannot build catalogue.")
        return pd.DataFrame()

    rows = []
    for r in regions:
        w    = r["w"]
        h    = r["h"]
        area = r["area"]

        aspect_ratio = round(w / h, 3) if h > 0 else 0.0

        bbox_area = w * h
        extent    = round(area / bbox_area, 3) if bbox_area > 0 else 0.0

        rows.append({
            "object_id":      int(r["id"]),
            "centroid_x":     r["cx"],
            "centroid_y":     r["cy"],
            "bbox_x":         int(r["x"]),
            "bbox_y":         int(r["y"]),
            "bbox_width":     int(w),
            "bbox_height":    int(h),
            "area_px":        int(area),
            "mean_intensity": r["mean_intensity"],
            "aspect_ratio":   aspect_ratio,
            "extent":         extent,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("area_px", ascending=False).reset_index(drop=True)

    print(f"[Stage 7] Catalogue built: {len(df)} candidate objects.")
    return df

def print_catalogue_summary(df: pd.DataFrame) -> None:
    """
    Print a readable summary of the catalogue to the terminal.

    Shows:
        - Overall statistics (count, area range, intensity range)
        - Top 20 rows by area
        - Distribution statistics for all numeric columns
    """
    if df.empty:
        print("[Stage 7] Catalogue is empty.")
        return

    print("\n" + "=" * 70)
    print("  OBJECT CATALOGUE SUMMARY")
    print("=" * 70)
    print(f"  Total candidate objects  : {len(df)}")
    print(f"  Area range (px)          : {df['area_px'].min()} – {df['area_px'].max()}")
    print(f"  Mean intensity range     : {df['mean_intensity'].min():.1f} – "
          f"{df['mean_intensity'].max():.1f}")
    print(f"  Aspect ratio range       : {df['aspect_ratio'].min():.2f} – "
          f"{df['aspect_ratio'].max():.2f}")
    print(f"  Extent range             : {df['extent'].min():.3f} – "
          f"{df['extent'].max():.3f}")

    print("\n--- Top 20 candidate objects by area ---")
    # Format display columns
    display_cols = ["object_id", "centroid_x", "centroid_y",
                    "area_px", "mean_intensity", "aspect_ratio", "extent"]
    print(df[display_cols].head(20).to_string(index=False))

    print("\n--- Descriptive statistics (all numeric columns) ---")
    print(df.describe().round(2).to_string())
    print("=" * 70)

def display_catalogue_plots(df: pd.DataFrame) -> None:

    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stage 7 — Object Catalogue Feature Distributions", fontsize=14)

    # Panel 1: Scatter — Area vs Mean Intensity
    ax = axes[0][0]
    ax.scatter(df["area_px"], df["mean_intensity"],
               s=8, alpha=0.5, color="royalblue", edgecolors="none")
    ax.set_xlabel("Area (pixels)")
    ax.set_ylabel("Mean Intensity")
    ax.set_title("Area vs Mean Intensity")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)

    # Panel 2: Area histogram (log x-axis)
    ax = axes[0][1]
    ax.hist(df["area_px"], bins=60, color="steelblue",
            edgecolor="none", alpha=0.85)
    ax.set_xlabel("Area (pixels, log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Area Distribution")
    ax.set_xscale("log")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Aspect Ratio histogram
    ax = axes[1][0]
    ax.hist(df["aspect_ratio"], bins=40, color="mediumseagreen",
            edgecolor="none", alpha=0.85)
    ax.set_xlabel("Aspect Ratio (width / height)")
    ax.set_ylabel("Count")
    ax.set_title("Aspect Ratio Distribution\n"
                 "(~1.0 = compact,  >2 or <0.5 = elongated)")
    ax.axvline(1.0, color="red", linewidth=1.2,
               linestyle="--", label="Aspect = 1.0")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Extent histogram
    ax = axes[1][1]
    ax.hist(df["extent"], bins=40, color="coral",
            edgecolor="none", alpha=0.85)
    ax.set_xlabel("Extent  (area / bbox area)")
    ax.set_ylabel("Count")
    ax.set_title("Extent Distribution\n"
                 "(~1.0 = point source,  <0.7 = extended/irregular)")
    ax.axvline(0.785, color="navy", linewidth=1.2,
               linestyle="--", label="π/4 ≈ 0.785 (filled disk)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

def display_annotated_image(gray: np.ndarray,
                             df: pd.DataFrame,
                             image_path: str = "",
                             top_n: int = 30) -> None:

    if df.empty:
        return

    fname = os.path.basename(image_path) if image_path else "image"
    top_df = df.head(top_n)   # already sorted by area descending

    fig, ax = plt.subplots(1, 1, figsize=(13, 11))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)

    for _, row in top_df.iterrows():
        x, y  = int(row["bbox_x"]),    int(row["bbox_y"])
        w, h  = int(row["bbox_width"]), int(row["bbox_height"])
        cx, cy = float(row["centroid_x"]), float(row["centroid_y"])
        oid   = int(row["object_id"])

        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=1.0, edgecolor="yellow", facecolor="none", alpha=0.9
        )
        ax.add_patch(rect)

        # Label: object_id above the bounding box
        ax.text(x, y - 4, f"#{oid}",
                color="yellow", fontsize=6, fontweight="bold",
                va="bottom", ha="left",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        # Centroid dot
        ax.plot(cx, cy, ".", color="red", markersize=3, alpha=0.9)

    ax.set_title(
        f"Stage 7 — Top {top_n} Candidate Objects by Area  ({fname})\n"
        f"Yellow box = bounding box  |  Red dot = centroid  |  #ID = object_id",
        fontsize=10, pad=8
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def export_catalogue(df: pd.DataFrame,
                     output_path: str = None) -> str:

    if df.empty:
        print("[Stage 7] Catalogue is empty — nothing to export.")
        return ""

    if output_path is None:
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "object_catalogue.csv")

    df.to_csv(output_path, index=False)
    print(f"\n[Stage 7] Catalogue exported to:\n  {output_path}")
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")
    return output_path


# ---------------------------------------------------------------------------
# Stage 7 entry point
# ---------------------------------------------------------------------------
def run_stage7(regions: list,
               gray: np.ndarray,
               image_path: str = "",
               save_csv: bool = True,
               csv_path: str = None) -> pd.DataFrame:

    print("=" * 60)
    print("  STAGE 7: FEATURE EXTRACTION AND OBJECT CATALOGUE")
    print("=" * 60)

    df = build_catalogue(regions)
    if df.empty:
        return df

    print_catalogue_summary(df)

    display_catalogue_plots(df)

    display_annotated_image(gray, df, image_path, top_n=30)

    # Step E: Export to CSV
    if save_csv:
        export_catalogue(df, output_path=csv_path)

    print("\n[Stage 7 complete]")
    print("Next step -> Stage 8 (final comparison visualisation) when ready.\n")

    return df

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
        from stage6_objects      import run_stage6

        original_image, _ = run_stage1(image_path)
        gray_image        = run_stage2(original_image, image_path)
        enhanced_image    = run_stage3(gray_image, image_path)
        filter_results    = run_stage4(enhanced_image, image_path, kernel_size=3)
        median_image      = filter_results["median"]
        seg_results       = run_stage5(median_image, image_path,
                                       global_t=130, block_size=51, C=5)
        otsu_mask         = seg_results["mask_otsu"]
        regions           = run_stage6(otsu_mask, enhanced_image,
                                       image_path, min_area=5,
                                       remove_border=True)

    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)

    catalogue_df = run_stage7(
        regions    = regions,
        gray       = enhanced_image,
        image_path = image_path,
        save_csv   = True,
    )
