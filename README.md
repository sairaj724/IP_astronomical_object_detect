# Astronomical Image Processing — Classical DIP Pipeline

An end-to-end Digital Image Processing (DIP) project for astronomical deep-field observation and analysis. 

The pipeline uses **pure classical image-processing algorithms** (no machine learning or deep learning) to enhance faint celestial bodies, suppress cosmic sensor noise, segment candidate stars and galaxies, and automatically extract morphological and photometric features into an astronomical catalogue.

---

## 🌌 Project Architecture & Stages

```
IP project/
├── stage1_load_inspect.py   # Stage 1: Load image & inspect intensity distributions
├── stage2_grayscale.py      # Stage 2: Luminance conversion (ITU-R BT.601)
├── stage3_contrast.py       # Stage 3: Percentile-based contrast enhancement
├── stage4_filtering.py      # Stage 4: Denoising (Gaussian & Median filtering)
├── stage5_segmentation.py   # Stage 5: Bimodal Otsu & Adaptive thresholding
├── stage6_objects.py        # Stage 6: Connected-Component Labelling (8-connectivity)
├── stage7_catalogue.py      # Stage 7: Feature extraction & CSV catalogue generation
├── stage8_visualise.py      # Stage 8: Unified master pipeline comparison figure
│
├── ui/                      # Interactive Web GUI (Observatory Dashboard)
│   ├── app.py               # Flask application server
│   ├── static/              # CSS styles & client-side scripts
│   └── templates/           # HTML dashboard interface
│
├── outputs/                 # Auto-saved figures and exported CSV catalogue
│   ├── object_catalogue.csv # 1,150+ detected candidate objects
│   └── stage8_pipeline_comparison.png
│
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules for builds, pycache & temp files
└── README.md                # Project documentation & usage guide
```

---

## 🛠️ Installation

Ensure you have **Python 3.9+** installed.

1. Open your terminal in the project directory:
   ```bash
   cd "c:\path\to\IP project"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Use This Project

### Method 1: Interactive Web Dashboard (Recommended)

Run the deep-space observatory dashboard in your browser:

1. Start the Flask server:
   ```bash
   python ui/app.py
   ```
2. Open your web browser and go to:
   ```
   http://127.0.0.1:5000
   ```
3. **Features in the UI:**
   - **Upload Image:** Drag-and-drop any astronomical PNG/JPG or use the default deep-field frame.
   - **Interactive Parameter Tuning:** Adjust contrast percentiles, filter kernels, threshold methods (Otsu vs Adaptive), and minimum object area.
   - **Live Pipeline Grid:** View side-by-side results across all processing stages.
   - **Interactive Object Inspector:** Click on any candidate object to view a zoomed crop with its centroid, area, aspect ratio, and mean brightness.
   - **One-Click CSV Export:** Download the updated catalogue directly from the browser.

---

### Method 2: Command-Line & Master Figure Generation

#### 1. Generate the Complete Master Report Figure (Stage 8)
Runs Stages 1 through 7 sequentially, saves the comparison figure, and exports the catalogue CSV:
```bash
python stage8_visualise.py
```
*Outputs generated:*
- `outputs/stage8_pipeline_comparison.png`
- `outputs/object_catalogue.csv`

#### 2. Run Any Stage Individually
You can test and inspect any single stage in isolation:

| Stage | Command | Purpose |
|---|---|---|
| **Stage 1** | `python stage1_load_inspect.py` | Validates bit-depth, channels, and intensity statistics |
| **Stage 2** | `python stage2_grayscale.py` | Converts BGR $\rightarrow$ Grayscale using ITU-R BT.601 |
| **Stage 3** | `python stage3_contrast.py` | Compares Min-Max vs Percentile contrast stretching |
| **Stage 4** | `python stage4_filtering.py` | Compares Gaussian blur vs Median denoising |
| **Stage 5** | `python stage5_segmentation.py` | Compares Global, Otsu, and Adaptive threshold masks |
| **Stage 6** | `python stage6_objects.py` | Groups pixels via 8-connected CCL & filters borders |
| **Stage 7** | `python stage7_catalogue.py` | Extracts 11 features per object into a Pandas DataFrame |

*(Note: Matplotlib will display interactive plot windows. Close each window to proceed to the next step).*

---

## 💡 Useful Tips & Best Practices

### 1. Parameter Tuning Guide
- **Too many tiny speckles detected?**
  - Increase `min_area` in Stage 6 (e.g. from `5` to `15` or `20` pixels). This filters out single-pixel cosmic ray hits.
- **Faint galaxy arms getting cut off?**
  - Check Stage 3 contrast stretch: use `p_low=1.0` and `p_high=99.0` to preserve the outer diffuse halo.
  - In Stage 5, Otsu thresholding balances foreground/background automatically. If objects disappear, lower the manual threshold.
- **Diffraction spikes and stars merged?**
  - Bright stars with diffraction spikes touch neighboring galaxies in the binary mask. In classical DIP, this is a known limitation of Connected-Component Labelling. The theoretical solution is the **Watershed Transform** (Distance Transform + marker-controlled flooding).

### 2. Handling New Astronomical Images
You can pass custom image paths directly as a command-line argument to any stage:
```bash
python stage1_load_inspect.py "C:\path\to\your_image.png"
python stage8_visualise.py "C:\path\to\your_image.png"
```

### 3. Understanding Catalogue Output (`outputs/object_catalogue.csv`)
Each row corresponds to one detected candidate celestial object:
- `object_id`: Unique integer assigned by Connected-Component Labelling.
- `centroid_x`, `centroid_y`: Exact center of mass (in pixels).
- `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`: Bounding box coordinates.
- `area_px`: Number of foreground pixels in the region.
- `mean_intensity`: Average grayscale brightness of the object.
- `aspect_ratio`: $\frac{\text{width}}{\text{height}}$ ($\approx 1.0$ indicates compact/circular; $> 2.0$ or $< 0.5$ suggests elongated edge-on galaxies).
- `extent`: $\frac{\text{area}}{\text{bbox\_width} \times \text{bbox\_height}}$ (compactness indicator; $\approx 0.785$ for filled circular disks).

---

