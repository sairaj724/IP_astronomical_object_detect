"""
Cosmic Astronomical Image Processing - Flask Web UI
Serves Stage 8 (Pipeline Visualizer) and Stage 9 (Object Inspector / Catalogue)
"""

import os
import sys
import io
import base64
import traceback

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from flask import Flask, render_template, request, jsonify, send_file

# Add the parent project directory to path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

from stage1_load_inspect import load_image
from stage2_grayscale    import convert_to_grayscale
from stage3_contrast     import percentile_stretch
from stage4_filtering    import apply_gaussian_filter, apply_median_filter
from stage5_segmentation import otsu_threshold, global_threshold, adaptive_threshold
from stage6_objects      import run_connected_components, filter_regions, measure_intensity
from stage7_catalogue    import build_catalogue, export_catalogue

app = Flask(__name__, template_folder="templates", static_folder="static")

DEFAULT_IMAGE = r"C:\Users\saira\Downloads\ip-req\img1.png"
UPLOAD_FOLDER = os.path.join(PROJECT_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def _encode_image(img, cmap="gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#0b0e14")
    ax.set_facecolor("#0b0e14")
    if img.ndim == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def _encode_overlay(gray, regions, top_n=1000):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0b0e14")
    ax.set_facecolor("#0b0e14")
    ax.imshow(gray, cmap="gray")
    display_regions = sorted(regions, key=lambda r: r.get("area", 0), reverse=True)[:top_n]
    for r in display_regions:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cx, cy = r["cx"], r["cy"]
        area = r["area"]
        col = "#00e5ff" if area > 500 else ("#ffd600" if area > 50 else "#ff4081")
        rect = patches.Rectangle((x, y), w, h, linewidth=0.8,
                                  edgecolor=col, facecolor="none", alpha=0.9)
        ax.add_patch(rect)
        ax.plot(cx, cy, "+", color="#ffffff", markersize=2.5, markeredgewidth=0.6)
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def _encode_scatter(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    sc = ax.scatter(
        df["area_px"], df["mean_intensity"],
        c=df["extent"], cmap="plasma",
        s=np.clip(df["area_px"] / 30, 2, 80), alpha=0.75, linewidths=0
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Extent", color="#90caf9", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#90caf9")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#90caf9")
    ax.set_xscale("log")
    ax.set_xlabel("Area (px) - log scale", color="#90caf9", fontsize=10)
    ax.set_ylabel("Mean Intensity", color="#90caf9", fontsize=10)
    ax.set_title("Stage 9 - Object Feature Space", color="#64b5f6", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#90caf9")
    for spine in ax.spines.values():
        spine.set_color("#2a3b5c")
    ax.grid(color="#1e2d45", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def run_pipeline(image_path, params):
    img     = load_image(image_path)
    gray, _ = convert_to_grayscale(img)
    p_low   = float(params.get("p_low", 2.0))
    p_high  = float(params.get("p_high", 98.0))
    enhanced = percentile_stretch(gray, p_low=p_low, p_high=p_high)

    filter_type = params.get("filter_type", "median")
    kernel_size = int(params.get("kernel_size", 3))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if filter_type == "gaussian":
        sigma    = float(params.get("sigma", 0))
        filtered = apply_gaussian_filter(enhanced, kernel_size=kernel_size, sigma=sigma)
    else:
        filtered = apply_median_filter(enhanced, kernel_size=kernel_size)

    thresh_method = params.get("thresh_method", "otsu")
    if thresh_method == "global":
        t_val    = int(params.get("global_t", 130))
        mask     = global_threshold(filtered, threshold=t_val)
        otsu_val = t_val
    elif thresh_method == "adaptive":
        block = int(params.get("block_size", 51))
        C_val = int(params.get("C_val", 5))
        if block % 2 == 0:
            block += 1
        mask     = adaptive_threshold(filtered, block_size=block, C=C_val)
        otsu_val = 0
    else:
        mask, otsu_val = otsu_threshold(filtered)
        otsu_val = int(otsu_val)

    min_area = int(params.get("min_area", 5))
    max_area = params.get("max_area", None)
    max_area = int(max_area) if max_area else None

    num_labels, labels, stats, centroids = run_connected_components(mask)
    regions = filter_regions(num_labels, stats, centroids,
                             img.shape[:2], min_area=min_area,
                             max_area=max_area, remove_border=True)
    regions = measure_intensity(regions, labels, enhanced)
    df = build_catalogue(regions)

    return {
        "img": img, "gray": gray, "enhanced": enhanced,
        "filtered": filtered, "mask": mask,
        "regions": regions, "df": df, "otsu_val": otsu_val,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, "current_image.png")
    f.save(save_path)
    return jsonify({"status": "ok", "path": save_path})


@app.route("/api/run_stage8", methods=["POST"])
def run_stage8_api():
    try:
        body       = request.get_json(force=True) or {}
        image_path = body.get("image_path", os.path.join(UPLOAD_FOLDER, "current_image.png"))
        if not os.path.exists(image_path):
            image_path = DEFAULT_IMAGE
        params = body.get("params", {})
        result = run_pipeline(image_path, params)
        panels = {
            "original": _encode_image(result["img"]),
            "gray":     _encode_image(result["gray"],     cmap="gray"),
            "enhanced": _encode_image(result["enhanced"], cmap="magma"),
            "filtered": _encode_image(result["filtered"], cmap="gray"),
            "mask":     _encode_image(result["mask"],     cmap="bone"),
            "overlay":  _encode_overlay(result["enhanced"], result["regions"]),
        }
        return jsonify({
            "panels":      panels,
            "n_objects":   len(result["regions"]),
            "otsu_val":    result["otsu_val"],
            "image_shape": list(result["gray"].shape),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_stage9", methods=["POST"])
def run_stage9_api():
    try:
        body       = request.get_json(force=True) or {}
        image_path = body.get("image_path", os.path.join(UPLOAD_FOLDER, "current_image.png"))
        if not os.path.exists(image_path):
            image_path = DEFAULT_IMAGE
        params = body.get("params", {})
        result = run_pipeline(image_path, params)
        df     = result["df"]

        min_intensity = float(body.get("min_intensity", 0))
        max_intensity = float(body.get("max_intensity", 255))
        min_area_f    = int(body.get("min_area_filter", 0))
        max_area_f    = int(body.get("max_area_filter", 999999))

        mask_df = (
            (df["mean_intensity"] >= min_intensity) &
            (df["mean_intensity"] <= max_intensity) &
            (df["area_px"] >= min_area_f) &
            (df["area_px"] <= max_area_f)
        )
        filtered_df = df[mask_df].copy()

        scatter_b64      = _encode_scatter(filtered_df) if not filtered_df.empty else ""
        filtered_ids     = set(filtered_df["object_id"].tolist())
        filtered_regions = [r for r in result["regions"] if r["id"] in filtered_ids]
        overlay_b64      = _encode_overlay(result["enhanced"], filtered_regions)
        catalogue_records = filtered_df.head(500).to_dict(orient="records")

        return jsonify({
            "overlay":    overlay_b64,
            "scatter":    scatter_b64,
            "catalogue":  catalogue_records,
            "n_filtered": len(filtered_df),
            "n_total":    len(df),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/export_csv", methods=["POST"])
def export_csv_api():
    try:
        body       = request.get_json(force=True) or {}
        image_path = body.get("image_path", os.path.join(UPLOAD_FOLDER, "current_image.png"))
        if not os.path.exists(image_path):
            image_path = DEFAULT_IMAGE
        params   = body.get("params", {})
        result   = run_pipeline(image_path, params)
        csv_path = os.path.join(OUTPUT_FOLDER, "object_catalogue.csv")
        export_catalogue(result["df"], csv_path)
        return send_file(csv_path, as_attachment=True, download_name="object_catalogue.csv")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/click_object", methods=["POST"])
def click_object():
    try:
        body       = request.get_json(force=True) or {}
        image_path = body.get("image_path", os.path.join(UPLOAD_FOLDER, "current_image.png"))
        if not os.path.exists(image_path):
            image_path = DEFAULT_IMAGE
        params  = body.get("params", {})
        click_x = float(body.get("click_x", 0))
        click_y = float(body.get("click_y", 0))
        result  = run_pipeline(image_path, params)
        regions = result["regions"]
        if not regions:
            return jsonify({"found": False})
        best, best_d = None, float("inf")
        for r in regions:
            d = (r["cx"] - click_x) ** 2 + (r["cy"] - click_y) ** 2
            if d < best_d:
                best_d, best = d, r
        if best_d > 50000:
            return jsonify({"found": False})
        df_row = result["df"][result["df"]["object_id"] == best["id"]].to_dict(orient="records")
        info   = df_row[0] if df_row else best
        return jsonify({"found": True, "object": info, "distance_px": float(best_d ** 0.5)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  COSMIC ASTRONOMICAL IMAGE PROCESSING - WEB UI")
    print("  Open  http://127.0.0.1:5000  in your browser")
    print("=" * 60)
    app.run(debug=False, port=5000)
