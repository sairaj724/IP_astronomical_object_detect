import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def _sep(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)

def load_image(image_path: str) -> np.ndarray:

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"\n[ERROR] No file found at:\n  {image_path}\n"
            "Please check the path and try again."
        )

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(
            f"\n[ERROR] OpenCV could not read the file:\n  {image_path}\n"
            "The file may be corrupt or in an unsupported format.\n"
            "Supported formats include: PNG, JPG, BMP, TIFF."
        )

    print(f"[OK] Image loaded successfully from:\n     {image_path}")
    return image


def inspect_image(image: np.ndarray) -> dict:

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
        colour_mode = "Grayscale"
    elif image.ndim == 3:
        height, width, channels = image.shape
        if channels == 3:
            colour_mode = "Color (BGR)"
        elif channels == 4:
            colour_mode = "Color (BGRA - includes alpha channel)"
        else:
            colour_mode = f"Unknown ({channels} channels)"
    else:
        raise ValueError(f"Unexpected image dimensionality: {image.ndim}D array.")

    dtype      = str(image.dtype)
    min_val    = int(image.min())
    max_val    = int(image.max())
    mean_val   = float(image.mean())

    _sep("IMAGE INSPECTION REPORT")
    print(f"  File dimensions : {height} rows x {width} columns")
    print(f"  Channels        : {channels}  ->  {colour_mode}")
    print(f"  Data type       : {dtype}")
    print(f"  Intensity range : min = {min_val},  max = {max_val}")
    print(f"  Mean intensity  : {mean_val:.2f}")

    if dtype == "uint8":
        full_range = 255
    elif dtype == "uint16":
        full_range = 65535
    else:
        full_range = None  # float images: range is not fixed

    if full_range is not None:
        used_pct = (max_val - min_val) / full_range * 100
        print(f"  Dynamic range   : {max_val - min_val} / {full_range}  "
              f"({used_pct:.1f}% of available range)")
        if used_pct < 30:
            print("  [!] Low dynamic range - contrast enhancement is recommended.")
        else:
            print("  [OK] Dynamic range appears reasonable.")

    _sep()

    info = {
        "height": height,
        "width": width,
        "channels": channels,
        "colour_mode": colour_mode,
        "dtype": dtype,
        "min_intensity": min_val,
        "max_intensity": max_val,
        "mean_intensity": round(mean_val, 2),
    }
    return info


def display_original(image: np.ndarray, image_path: str) -> None:

    title = f"Stage 1 - Original Image\n({os.path.basename(image_path)})"

    plt.figure(figsize=(8, 6))

    if image.ndim == 2:
       
        plt.imshow(image, cmap="gray", vmin=image.min(), vmax=image.max())
        plt.colorbar(label="Pixel intensity")
    elif image.ndim == 3 and image.shape[2] == 3:
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
    elif image.ndim == 3 and image.shape[2] == 4:
        
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        plt.imshow(rgba)
    else:
        plt.imshow(image, cmap="gray")

    plt.title(title, fontsize=13, pad=10)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def run_stage1(image_path: str):

    _sep("STAGE 1: IMAGE LOADING & INSPECTION")

    image = load_image(image_path)

    info = inspect_image(image)

    display_original(image, image_path)

    print("\n[Stage 1 complete]")
    print("Next step -> run Stage 2 (grayscale conversion) when ready.\n")

    return image, info

if __name__ == "__main__":

    image_path = r"C:\Users\saira\Downloads\ip-req\img1.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    try:
        original_image, image_info = run_stage1(image_path)
    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)
