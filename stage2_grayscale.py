import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grayscale(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image.ndim == 2:
        print("[Stage 2] Image is already grayscale. No conversion needed.")
        return image.copy(), False

    elif image.ndim == 3 and image.shape[2] == 4:
        # BGRA: drop the alpha channel first, then convert
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        print("[Stage 2] BGRA image converted to grayscale (alpha channel discarded).")
        return gray, True

    elif image.ndim == 3 and image.shape[2] == 3:
        # Standard BGR colour image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("[Stage 2] BGR colour image converted to grayscale.")
        print("          Formula used:  Y = 0.299*R + 0.587*G + 0.114*B")
        return gray, True

    else:
        raise ValueError(
            f"[Stage 2 ERROR] Unsupported image shape: {image.shape}. "
            "Expected (H,W), (H,W,3), or (H,W,4)."
        )

def report_grayscale(gray: np.ndarray) -> None:
 
    print("\n--- Grayscale Image Statistics ---")
    print(f"  Shape     : {gray.shape}  (H x W, single channel)")
    print(f"  dtype     : {gray.dtype}")
    print(f"  Min       : {gray.min()}")
    print(f"  Max       : {gray.max()}")
    print(f"  Mean      : {gray.mean():.2f}")
    print(f"  Std dev   : {gray.std():.2f}")

    if gray.std() < 15:
        print("  [Note] Low standard deviation — image has low contrast overall.")
        print("         Contrast enhancement in Stage 3 is recommended.")
    print("----------------------------------\n")

def display_grayscale(original: np.ndarray,
                      gray: np.ndarray,
                      was_converted: bool,
                      image_path: str = "") -> None:

    fname = os.path.basename(image_path) if image_path else "image"

    if not was_converted:
        # Single panel — already grayscale
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Stage 2 — Already Grayscale\n({fname})", fontsize=12)
        ax.axis("off")
    else:
        # Two panels — before and after
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Left: original colour image (convert BGR -> RGB for display)
        if original.ndim == 3 and original.shape[2] >= 3:
            display_orig = cv2.cvtColor(
                original[:, :, :3], cv2.COLOR_BGR2RGB
            )
            axes[0].imshow(display_orig)
        else:
            axes[0].imshow(original, cmap="gray")
        axes[0].set_title(f"Original\n({fname})", fontsize=11)
        axes[0].axis("off")

        # Right: grayscale result
        axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Stage 2 — Grayscale Conversion\n(Y = 0.299R + 0.587G + 0.114B)",
                           fontsize=11)
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def run_stage2(original_image: np.ndarray, image_path: str = "") -> np.ndarray:

    print("=" * 60)
    print("  STAGE 2: GRAYSCALE CONVERSION")
    print("=" * 60)

    gray, was_converted = convert_to_grayscale(original_image)

    report_grayscale(gray)

    display_grayscale(original_image, gray, was_converted, image_path)

    print("[Stage 2 complete]")
    print("Next step -> Stage 3 (contrast enhancement) when ready.\n")

    return gray

if __name__ == "__main__":
    import sys

    image_path = r"C:\Users\saira\Downloads\ip-req\img1.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    try:
        from stage1_load_inspect import run_stage1
        original_image, _ = run_stage1(image_path)
    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)

    # Run Stage 2
    grayscale_image = run_stage2(original_image, image_path)
